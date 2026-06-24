from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


def canonical_source(source: str) -> str:
    return source.removeprefix("shape.")


@dataclass(frozen=True)
class ShapeFacts:
    shape: Mapping[str, int]

    def value(self, *names: str, default: int | None = None) -> int:
        for name in names:
            if name in self.shape:
                return int(self.shape[name])
        if default is None:
            raise KeyError(names[0])
        return default

    @property
    def ncols(self) -> int:
        return self.value("ncols", "cols", default=1)

    @property
    def nrows(self) -> int:
        return self.value("nrows", "rows", default=1)

    @property
    def rows(self) -> int:
        return self.value("rows", "nrows", default=1)

    @property
    def cols(self) -> int:
        return self.value("cols", "ncols", default=1)

    @property
    def k(self) -> int:
        return self.value("k", "ncols", "cols", default=1)

    @property
    def n_dims(self) -> int:
        return self.value("n_dims", default=min(self.ncols, 128))

    @property
    def element_count(self) -> int:
        return self.ncols * self.nrows


class Expr(Protocol):
    def resolve(self, facts: ShapeFacts) -> int: ...


@dataclass(frozen=True)
class Const:
    value: int

    def resolve(self, facts: ShapeFacts) -> int:
        return self.value


@dataclass(frozen=True)
class ShapeValue:
    names: tuple[str, ...]
    default: int | None = None

    def resolve(self, facts: ShapeFacts) -> int:
        return facts.value(*self.names, default=self.default)


@dataclass(frozen=True)
class Product:
    terms: tuple[Expr, ...]

    def resolve(self, facts: ShapeFacts) -> int:
        value = 1
        for term in self.terms:
            value *= term.resolve(facts)
        return value


@dataclass(frozen=True)
class Maximum:
    terms: tuple[Expr, ...]

    def resolve(self, facts: ShapeFacts) -> int:
        return max(term.resolve(facts) for term in self.terms)


@dataclass(frozen=True)
class Minimum:
    terms: tuple[Expr, ...]

    def resolve(self, facts: ShapeFacts) -> int:
        return min(term.resolve(facts) for term in self.terms)


@dataclass(frozen=True)
class CeilDiv:
    numerator: Expr
    denominator: int

    def resolve(self, facts: ShapeFacts) -> int:
        return math.ceil(self.numerator.resolve(facts) / self.denominator)


@dataclass(frozen=True)
class FamilySpec:
    family_ids: tuple[str, ...]
    bindings: Mapping[str, Expr]

    def resolve(self, source: str, shape: Mapping[str, int]) -> int | None:
        key = canonical_source(source)
        expr = self.bindings.get(key)
        if expr is None:
            return None
        return expr.resolve(ShapeFacts(shape))


DEFAULT_AXIS_VALUES: dict[str, tuple[int, ...]] = {
    "k": (256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 11008, 14336, 16384, 28672, 32768),
    "rows": (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, 16384, 32768),
    "cols": (1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
    "ncols": (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 11008, 32768, 65536),
    "nrows": (1, 2, 4, 8, 16, 32, 60, 64, 128, 256, 512, 1024, 4096, 16384, 1048576),
    "n_dims": (32, 64, 80, 96, 128),
}


@dataclass(frozen=True)
class ShapeDomain:
    family: str
    route_id: str
    root_symbol: str
    domain: Mapping[str, Any]
    guards: Mapping[str, Any]

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(name for name in DEFAULT_AXIS_VALUES if self.has_axis(name))

    def has_axis(self, name: str) -> bool:
        return f"{name}_min" in self.domain or f"{name}_max" in self.domain

    def bounds(self, name: str) -> tuple[int, int]:
        defaults = DEFAULT_AXIS_VALUES[name]
        lo_raw = self.domain.get(f"{name}_min")
        hi_raw = self.domain.get(f"{name}_max")
        lo = int(lo_raw if lo_raw is not None else min(defaults))
        hi = int(hi_raw if hi_raw is not None else max(defaults))
        return lo, hi

    def multiple(self, name: str) -> int | None:
        multiple = self.guards.get(f"{name}_multiple_of")
        if not multiple:
            return None
        multiple_i = int(multiple)
        return multiple_i if multiple_i > 1 else None

    def align(self, name: str, value: int, *, direction: str) -> int:
        multiple = self.multiple(name)
        if not multiple:
            return value
        if direction == "up":
            return int(math.ceil(value / multiple) * multiple)
        return int(math.floor(value / multiple) * multiple)

    def choose(self, name: str, preferred: tuple[int, ...] = (), *, fallback: str = "lo") -> int | None:
        if not self.has_axis(name):
            return None
        lo, hi = self.bounds(name)
        multiple = self.multiple(name)
        for value in preferred:
            if lo <= value <= hi and (not multiple or value % multiple == 0):
                return value
        if fallback == "hi":
            value = self.align(name, hi, direction="down")
        elif fallback == "mid":
            mid = (lo + hi) // 2
            value = self.align(name, mid, direction="down")
            if value < lo:
                value = self.align(name, mid, direction="up")
        else:
            value = self.align(name, lo, direction="up")
        return value if lo <= value <= hi else None

    def point(self, preferences: Mapping[str, tuple[int, ...]], *, fallback: str = "lo") -> dict[str, int] | None:
        shape: dict[str, int] = {}
        for axis in self.axes:
            value = self.choose(axis, preferences.get(axis, ()), fallback=fallback)
            if value is None:
                return None
            shape[axis] = value
        return normalize_shape(shape)

    def accepts(self, shape: Mapping[str, int]) -> bool:
        for axis in self.axes:
            if axis not in shape:
                return False
            value = int(shape[axis])
            lo, hi = self.bounds(axis)
            multiple = self.multiple(axis)
            if value < lo or value > hi:
                return False
            if multiple and value % multiple != 0:
                return False
        return True


ShapeProbe = Callable[[ShapeDomain], dict[str, int] | None]


@dataclass(frozen=True)
class SearchSchedule:
    family_ids: tuple[str, ...]
    edge_probes: tuple[ShapeProbe, ...]


def normalize_shape(shape: Mapping[str, int]) -> dict[str, int]:
    normalized = dict(shape)
    if "ncols" in normalized and "cols" not in normalized:
        normalized["cols"] = normalized["ncols"]
    if "nrows" in normalized and "rows" not in normalized:
        normalized["rows"] = normalized["nrows"]
    return normalized


def _dedupe_shapes(shapes: list[dict[str, int] | None]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for shape in shapes:
        if not shape:
            continue
        key = tuple(sorted(shape.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(shape)
    return result


def _smoke_probe(ctx: ShapeDomain) -> dict[str, int] | None:
    return ctx.point(
        {
            "k": (256, 512, 1024),
            "rows": (1, 64, 128),
            "cols": (1, 16, 64),
            "ncols": (64, 128, 256),
            "nrows": (1, 8, 16),
            "n_dims": (64, 80, 128),
        },
        fallback="lo",
    )


def _mid_probe(ctx: ShapeDomain) -> dict[str, int] | None:
    return ctx.point({}, fallback="mid")


def _max_probe(ctx: ShapeDomain) -> dict[str, int] | None:
    return ctx.point({}, fallback="hi")


def _matmul_prompt_probe(ctx: ShapeDomain) -> dict[str, int] | None:
    return ctx.point(
        {
            "k": (3072, 4096, 8192),
            "rows": (64, 128, 512, 3072),
            "cols": (64, 128, 256),
        },
        fallback="mid",
    )


def _matmul_decode_probe(ctx: ShapeDomain) -> dict[str, int] | None:
    return ctx.point(
        {
            "k": (3072, 4096, 8192),
            "rows": (1, 8, 16, 64),
            "cols": (1, 16, 32, 64),
        },
        fallback="lo",
    )


def _row_major_long_probe(ctx: ShapeDomain) -> dict[str, int] | None:
    return ctx.point(
        {
            "ncols": (4096, 8192, 11008, 32768),
            "nrows": (1, 8, 64),
            "cols": (4096, 8192, 11008, 32768),
            "rows": (1, 8, 64),
        },
        fallback="hi",
    )


def _token_batch_probe(ctx: ShapeDomain) -> dict[str, int] | None:
    return ctx.point(
        {
            "ncols": (64, 128, 256),
            "nrows": (64, 128, 512),
            "cols": (64, 128, 256),
            "rows": (64, 128, 512),
            "n_dims": (64, 80, 128),
        },
        fallback="mid",
    )


MUL_MAT_SEARCH = SearchSchedule(
    family_ids=(
        "mul_mat_q4_k_f32",
        "mul_mat_q5_k_f32",
        "mul_mat_q6_k_f32",
        "mul_mat_q8_0_f32",
        "mul_mat_f16_f32_batched",
        "mul_mat_f16_f32_batched_cont",
        "mul_mat_f16_f32_batched_kq_split_experiment",
        "mul_mat_id_q4_k_f32",
        "mul_mat_id_q5_k_f32",
        "mul_mat_id_q6_k_f32",
    ),
    edge_probes=(_smoke_probe, _matmul_decode_probe, _matmul_prompt_probe, _max_probe),
)


ROW_MAJOR_SEARCH = SearchSchedule(
    family_ids=(
        "copy_f32_f16",
        "cont_f32",
        "cont_set_rows_f32",
        "rms_norm_f32",
        "soft_max_f32",
        "rope_f32",
        "rope_neox_f32",
        "rope_scale_f32",
        "rope_set_rows_f32",
        "get_rows_f32",
        "get_rows_q4_k_f32",
        "get_rows_q5_k_f32",
        "get_rows_q6_k_f32",
        "get_rows_q8_0_f32",
    ),
    edge_probes=(_smoke_probe, _token_batch_probe, _row_major_long_probe),
)


GENERIC_SEARCH = SearchSchedule(
    family_ids=("__generic__",),
    edge_probes=(_smoke_probe, _mid_probe, _max_probe),
)


SEARCH_SCHEDULES = (MUL_MAT_SEARCH, ROW_MAJOR_SEARCH)
SEARCH_SCHEDULES_BY_FAMILY: dict[str, SearchSchedule] = {
    family_id: schedule
    for schedule in SEARCH_SCHEDULES
    for family_id in schedule.family_ids
}


def concrete_shapes_for_route(
    route: Mapping[str, Any],
    *,
    sweep: str,
    observed_shapes: Iterable[Mapping[str, int]] = (),
) -> list[dict[str, int]]:
    domain = dict(route.get("shape_domain") or {})
    if not domain:
        return [{}]
    ctx = ShapeDomain(
        family=str(route.get("family") or route.get("source_id") or "unknown"),
        route_id=str(route.get("id") or ""),
        root_symbol=str(route.get("root_symbol") or ""),
        domain=domain,
        guards=dict(route.get("shape_guards") or {}),
    )
    if not ctx.axes:
        return [default_shape_for_axisless_route(route)]
    if sweep == "observed":
        observed = [normalize_shape(shape) for shape in observed_shapes]
        accepted = [shape for shape in observed if ctx.accepts(shape)]
        return _dedupe_shapes(accepted or [_smoke_probe(ctx)])
    if sweep == "minimal":
        return _dedupe_shapes([_smoke_probe(ctx)])
    schedule = SEARCH_SCHEDULES_BY_FAMILY.get(ctx.family, GENERIC_SEARCH)
    return _dedupe_shapes([probe(ctx) for probe in schedule.edge_probes])


def default_shape_for_axisless_route(route: Mapping[str, Any]) -> dict[str, int]:
    family = str(route.get("family") or route.get("source_id") or "")
    defaults = {
        "quantize_q8_1_f32": {"ncols": 256, "nrows": 1, "cols": 256, "rows": 1},
    }
    return dict(defaults.get(family, {}))


N_COLS = ShapeValue(("ncols", "cols"), default=1)
N_ROWS = ShapeValue(("nrows", "rows"), default=1)
ROWS = ShapeValue(("rows", "nrows"), default=1)
COLS = ShapeValue(("cols", "ncols"), default=1)
K = ShapeValue(("k", "ncols", "cols"), default=1)
N_DIMS = ShapeValue(("n_dims",), default=128)
ONE = Const(1)
FOUR = Const(4)
EIGHT = Const(8)


COMMON_SPEC = FamilySpec(
    family_ids=("__common__",),
    bindings={
        "k": K,
        "rows": ROWS,
        "cols": COLS,
        "ncols": N_COLS,
        "nrows": N_ROWS,
        "n_dims": N_DIMS,
        "copy.n": Product((N_COLS, N_ROWS)),
        "q8_full_unroll_factor": ONE,
        "q8_1.blocks": Maximum((ONE, CeilDiv(N_COLS, 32))),
        "q8_1.ne1": N_ROWS,
        "q8_1.z_count": Maximum((ONE, N_ROWS)),
    },
)


POINTWISE_SPEC = FamilySpec(
    family_ids=("add_f32", "mul_f32", "div_f32", "clamp_f32", "scale_f32"),
    bindings={
        "pointwise.src0_row_stride": N_COLS,
        "pointwise.src1_row_stride": N_COLS,
        "pointwise.src1_ncols": N_COLS,
    },
)


ARGSORT_SPEC = FamilySpec(
    family_ids=("argsort_f32_i32",),
    bindings={
        "argsort.ncols": N_COLS,
        "argsort.nrows": N_ROWS,
    },
)


CONT_SPEC = FamilySpec(
    family_ids=("cont_f32", "cont_set_rows_f32"),
    bindings={
        "cont.ncols": N_COLS,
        "cont.nrows": N_ROWS,
        "cont.ne1": N_COLS,
        "cont.ne2": N_ROWS,
        "cont.src_nb1": Product((N_COLS, FOUR)),
        "cont.src_nb2": Product((N_COLS, N_ROWS, FOUR)),
        "cont.src_nb3": Product((N_COLS, N_ROWS, FOUR)),
    },
)


GET_ROWS_SPEC = FamilySpec(
    family_ids=("get_rows_f32", "get_rows_q4_k_f32", "get_rows_q5_k_f32", "get_rows_q6_k_f32", "get_rows_q8_0_f32"),
    bindings={
        "get_rows.ncols": N_COLS,
        "get_rows.nrows": N_ROWS,
        "get_rows.src0_nrows": Maximum((N_ROWS, ROWS, ONE)),
        "get_rows.idx_row_stride": ONE,
    },
)


GET_ROWS_MOE_SPEC = FamilySpec(
    family_ids=("get_rows_moe_weights_f32",),
    bindings={
        "get_rows_moe.nexperts": Maximum((ROWS, N_ROWS, ONE)),
        "get_rows_moe.nselected": Minimum((Maximum((COLS, ONE)), EIGHT)),
        "get_rows_moe.ntokens": Maximum((COLS, N_ROWS, ONE)),
        "get_rows_moe.src0_token_stride": Maximum((COLS, ONE)),
        "get_rows_moe.idx_token_stride": Maximum((COLS, ONE)),
        "get_rows_moe.dst_token_stride": Maximum((COLS, ONE)),
    },
)


SOFT_MAX_SPEC = FamilySpec(
    family_ids=("soft_max_f32",),
    bindings={
        "soft_max.ncols": N_COLS,
        "soft_max.nrows": N_ROWS,
        "soft_max.ne01": N_COLS,
        "soft_max.ne02": ONE,
        "soft_max.mask_ne1": N_ROWS,
        "soft_max.mask_ne2": ONE,
        "soft_max.mask_ne3": ONE,
        "soft_max.mask_nb1": Product((N_COLS, FOUR)),
        "soft_max.mask_nb2": Product((N_COLS, FOUR)),
        "soft_max.mask_nb3": Product((N_COLS, FOUR)),
    },
)


SWIGLU_SPEC = FamilySpec(
    family_ids=("swiglu_f32",),
    bindings={
        "swiglu.ncols": N_COLS,
        "swiglu.nrows": N_ROWS,
    },
)


ADD_RMS_NORM_MUL_SPEC = FamilySpec(
    family_ids=("add_rms_norm_mul_f32",),
    bindings={
        "add_rms_norm_mul.ncols": N_COLS,
        "add_rms_norm_mul.nrows": N_ROWS,
    },
)


RMS_NORM_MUL_SPEC = FamilySpec(
    family_ids=("rms_norm_mul_f32", "rms_norm_mul_quantize_q8_1_f32"),
    bindings={
        "rms_norm_mul.ncols": N_COLS,
        "rms_norm_mul.nrows": N_ROWS,
    },
)


ROPE_SPEC = FamilySpec(
    family_ids=("rope_f32", "rope_neox_f32", "rope_scale_f32", "rope_set_rows_f32"),
    bindings={
        "rope.ncols": N_COLS,
        "rope.nheads": ROWS,
        "rope.ntokens": Maximum((COLS, ONE)),
        "rope.n_dims": N_DIMS,
        "rope.src0_head_stride": N_COLS,
        "rope.src0_token_stride": Product((N_COLS, ROWS)),
        "rope.dst_head_stride": N_COLS,
        "rope.dst_token_stride": Product((N_COLS, ROWS)),
        "rope.pos_token_stride": ONE,
    },
)


MUL_MAT_F16_SPEC = FamilySpec(
    family_ids=("mul_mat_f16_f32_batched", "mul_mat_f16_f32_batched_cont", "mul_mat_f16_f32_batched_kq_split_experiment"),
    bindings={
        "mul_mat_f16.k": K,
        "mul_mat_f16.rows": ROWS,
        "mul_mat_f16.cols": COLS,
        "mul_mat_f16.dst_ne2": ONE,
        "mul_mat_f16.dst_ne3": ONE,
        "mul_mat_f16.src0_ne2": ONE,
        "mul_mat_f16.src0_ne3": ONE,
        "mul_mat_f16.src0_stride_row": K,
        "mul_mat_f16.src0_stride_ne2": Product((K, ROWS)),
        "mul_mat_f16.src0_stride_ne3": Product((K, ROWS)),
        "mul_mat_f16.src1_stride_col": K,
        "mul_mat_f16.src1_stride_ne2": Product((K, COLS)),
        "mul_mat_f16.src1_stride_ne3": Product((K, COLS)),
        "mul_mat_f16.dst_stride_col": ROWS,
        "mul_mat_f16.dst_stride_ne2": Product((ROWS, COLS)),
        "mul_mat_f16.dst_stride_ne3": Product((ROWS, COLS)),
    },
)


MUL_MAT_ID_SPEC = FamilySpec(
    family_ids=("mul_mat_id_q4_k_f32", "mul_mat_id_q5_k_f32", "mul_mat_id_q6_k_f32"),
    bindings={
        "mul_mat_id.k": K,
        "mul_mat_id.rows": ROWS,
        "mul_mat_id.nexperts": Maximum((N_ROWS, ROWS, ONE)),
        "mul_mat_id.nselected": Minimum((Maximum((COLS, ONE)), EIGHT)),
        "mul_mat_id.ntokens": Maximum((COLS, ONE)),
        "mul_mat_id.src1_selected_stride": K,
        "mul_mat_id.src1_token_stride": Maximum((ROWS, ONE)),
        "mul_mat_id.idx_token_stride": ONE,
        "mul_mat_id.dst_token_stride": Maximum((ROWS, ONE)),
    },
)


SET_ROWS_SPEC = FamilySpec(
    family_ids=("set_rows_f32", "cont_set_rows_f32", "rope_set_rows_f32"),
    bindings={
        "set_rows.nc": N_COLS,
        "set_rows.nr": N_ROWS,
        "set_rows.ne02": ONE,
        "set_rows.ne03": ONE,
        "set_rows.ne1": N_ROWS,
        "set_rows.ne11": ONE,
        "set_rows.ne12": ONE,
        "set_rows.src0_nb1": Product((N_COLS, FOUR)),
        "set_rows.src0_nb2": Product((N_COLS, N_ROWS, FOUR)),
        "set_rows.src0_nb3": Product((N_COLS, N_ROWS, FOUR)),
        "set_rows.idx_nb0": EIGHT,
        "set_rows.idx_nb1": Product((N_ROWS, EIGHT)),
        "set_rows.idx_nb2": Product((N_ROWS, EIGHT)),
        "set_rows.dst_nb1": Product((N_COLS, FOUR)),
        "set_rows.dst_nb2": Product((N_COLS, N_ROWS, FOUR)),
        "set_rows.dst_nb3": Product((N_COLS, N_ROWS, FOUR)),
    },
)


SUM_ROWS_SPEC = FamilySpec(
    family_ids=("sum_rows_f32",),
    bindings={
        "sum_rows.ncols": N_COLS,
        "sum_rows.nrows": N_ROWS,
        "sum_rows.src0_row_stride": N_COLS,
    },
)


FAMILY_SPECS = (
    POINTWISE_SPEC,
    ARGSORT_SPEC,
    CONT_SPEC,
    GET_ROWS_SPEC,
    GET_ROWS_MOE_SPEC,
    SOFT_MAX_SPEC,
    SWIGLU_SPEC,
    ADD_RMS_NORM_MUL_SPEC,
    RMS_NORM_MUL_SPEC,
    ROPE_SPEC,
    MUL_MAT_F16_SPEC,
    MUL_MAT_ID_SPEC,
    SET_ROWS_SPEC,
    SUM_ROWS_SPEC,
)

_SPECS_BY_FAMILY: dict[str, list[FamilySpec]] = {}
for _spec in FAMILY_SPECS:
    for _family_id in _spec.family_ids:
        _SPECS_BY_FAMILY.setdefault(_family_id, []).append(_spec)


def resolve_binding_value(family: str, source: str, shape: Mapping[str, int]) -> int | None:
    for spec in _SPECS_BY_FAMILY.get(family, []):
        value = spec.resolve(source, shape)
        if value is not None:
            return value
    return COMMON_SPEC.resolve(source, shape)
