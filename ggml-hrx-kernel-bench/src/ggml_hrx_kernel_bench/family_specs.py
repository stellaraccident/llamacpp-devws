from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol


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
        "soft_max.mask_nb2": Product((N_COLS, N_ROWS, FOUR)),
        "soft_max.mask_nb3": Product((N_COLS, N_ROWS, FOUR)),
    },
)


SWIGLU_SPEC = FamilySpec(
    family_ids=("swiglu_f32",),
    bindings={
        "swiglu.ncols": N_COLS,
        "swiglu.nrows": N_ROWS,
    },
)


ROPE_SPEC = FamilySpec(
    family_ids=("rope_f32", "rope_neox_f32", "rope_scale_f32", "rope_set_rows_f32"),
    bindings={
        "rope.ncols": N_COLS,
        "rope.nheads": Maximum((ROWS, N_ROWS, ONE)),
        "rope.ntokens": Maximum((COLS, ONE)),
        "rope.n_dims": N_DIMS,
        "rope.src0_head_stride": N_COLS,
        "rope.src0_token_stride": Product((N_COLS, Maximum((ROWS, N_ROWS, ONE)))),
        "rope.dst_head_stride": N_COLS,
        "rope.dst_token_stride": Product((N_COLS, Maximum((ROWS, N_ROWS, ONE)))),
        "rope.pos_token_stride": Product((N_COLS, Maximum((ROWS, N_ROWS, ONE)))),
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
