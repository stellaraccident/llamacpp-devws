// Local scratch probe: load a Loom-produced HSACO through public HRX APIs and
// dispatch a one-buffer kernel.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hrx_runtime.h"

static int fail_status(const char* operation, hrx_status_t status) {
  if (hrx_status_is_ok(status)) return 0;
  char* message = NULL;
  size_t length = 0;
  hrx_status_to_string(status, &message, &length);
  fprintf(stderr, "%s failed: %.*s\n", operation, (int)length,
          message ? message : "?");
  hrx_status_free_message(message);
  hrx_status_ignore(status);
  return 1;
}

static int check_status(const char* operation, hrx_status_t status) {
  if (hrx_status_is_ok(status)) return 0;
  return fail_status(operation, status);
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "/tmp/loom-hrx-targetless_store_i32.hsaco";
  const char* export_name = argc > 2 ? argv[2] : "targetless_store_i32";

  hrx_device_t device = NULL;
  hrx_executable_t executable = NULL;
  hrx_stream_t stream = NULL;
  hrx_buffer_t output = NULL;
  int result = 1;
  int gpu_initialized = 0;

  if (check_status("hrx_gpu_initialize", hrx_gpu_initialize(0))) goto cleanup;
  gpu_initialized = 1;
  if (check_status("hrx_gpu_device_get", hrx_gpu_device_get(0, &device))) {
    goto cleanup;
  }

  char architecture[64] = {0};
  if (check_status("hrx_device_get_property(ARCHITECTURE)",
                   hrx_device_get_property(device,
                                           HRX_DEVICE_PROPERTY_ARCHITECTURE,
                                           architecture,
                                           sizeof(architecture)))) {
    goto cleanup;
  }
  printf("hrx device[0] architecture=%s\n", architecture);

  if (check_status("hrx_executable_load_file",
                   hrx_executable_load_file(device, path, NULL,
                                            &executable))) {
    goto cleanup;
  }

  size_t export_count = 0;
  if (check_status("hrx_executable_export_count",
                   hrx_executable_export_count(executable, &export_count))) {
    goto cleanup;
  }
  printf("hrx loaded %s exports=%zu\n", path, export_count);
  for (size_t i = 0; i < export_count; ++i) {
    hrx_executable_export_info_t info = {0};
    if (check_status("hrx_executable_export_info",
                     hrx_executable_export_info(executable, (uint32_t)i,
                                                &info))) {
      goto cleanup;
    }
    printf("  export[%zu] name=%s constants=%u bindings=%u wg=%ux%ux%u\n", i,
           info.name ? info.name : "(null)", info.constant_byte_length,
           info.binding_count, info.workgroup_size[0], info.workgroup_size[1],
           info.workgroup_size[2]);
  }

  uint32_t export_ordinal = 0;
  if (check_status("hrx_executable_lookup_export_by_name",
                   hrx_executable_lookup_export_by_name(
                       executable, export_name, &export_ordinal))) {
    goto cleanup;
  }
  hrx_executable_export_info_t dispatch_info = {0};
  if (check_status("hrx_executable_export_info",
                   hrx_executable_export_info(executable, export_ordinal,
                                              &dispatch_info))) {
    goto cleanup;
  }
  if (dispatch_info.binding_count != 1 ||
      dispatch_info.constant_byte_length != 0) {
    fprintf(stderr,
            "unexpected export ABI for %s: constants=%u bindings=%u\n",
            export_name, dispatch_info.constant_byte_length,
            dispatch_info.binding_count);
    goto cleanup;
  }

  if (check_status("hrx_stream_create", hrx_stream_create(device, 0, &stream))) {
    goto cleanup;
  }
  if (check_status("hrx_buffer_allocate",
                   hrx_buffer_allocate(stream, sizeof(uint32_t),
                                       HRX_MEMORY_TYPE_DEVICE_LOCAL,
                                       HRX_BUFFER_USAGE_DEFAULT, &output))) {
    goto cleanup;
  }

  hrx_buffer_ref_t bindings[] = {{
      .buffer = output,
      .offset = 0,
      .length = sizeof(uint32_t),
  }};
  hrx_dispatch_config_t config = {
      .workgroup_count = {1, 1, 1},
      .workgroup_size =
          {
              dispatch_info.workgroup_size[0],
              dispatch_info.workgroup_size[1],
              dispatch_info.workgroup_size[2],
          },
      .subgroup_size = 0,
  };
  if (config.workgroup_size[0] == 0) config.workgroup_size[0] = 1;
  if (config.workgroup_size[1] == 0) config.workgroup_size[1] = 1;
  if (config.workgroup_size[2] == 0) config.workgroup_size[2] = 1;

  if (check_status("hrx_stream_dispatch",
                   hrx_stream_dispatch(stream, executable, export_ordinal,
                                       &config, NULL, 0, bindings, 1,
                                       HRX_DISPATCH_FLAG_NONE))) {
    goto cleanup;
  }
  if (check_status("hrx_stream_synchronize", hrx_stream_synchronize(stream))) {
    goto cleanup;
  }

  uint32_t actual = 0;
  if (check_status("hrx_synchronous_d2h",
                   hrx_synchronous_d2h(device, output, 0, &actual,
                                       sizeof(actual)))) {
    goto cleanup;
  }
  printf("hrx dispatched %s: output=%u\n", export_name, actual);
  result = actual == 42 ? 0 : 1;
  if (result != 0) {
    fprintf(stderr, "expected output=42\n");
  }

cleanup:
  hrx_buffer_release(output);
  hrx_stream_release(stream);
  hrx_executable_release(executable);
  if (gpu_initialized) {
    hrx_status_t shutdown_status = hrx_gpu_shutdown();
    if (!hrx_status_is_ok(shutdown_status)) {
      if (result == 0) result = 1;
      fail_status("hrx_gpu_shutdown", shutdown_status);
    }
  }
  return result;
}
