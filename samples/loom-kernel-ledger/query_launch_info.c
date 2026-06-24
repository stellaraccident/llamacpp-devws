// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "loomc/base.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/status.h"
#include "loomc/target.h"
#include "loomc/target/amdgpu/base.h"
#include "loomc/workspace.h"

static bool view_equals_symbol(loomc_string_view_t view, const char* symbol) {
  if (!symbol || !symbol[0]) return true;
  if (symbol[0] == '@') ++symbol;
  return strlen(symbol) == view.size && memcmp(symbol, view.data, view.size) == 0;
}

static void print_view_json(loomc_string_view_t view) {
  for (loomc_host_size_t i = 0; i < view.size; ++i) {
    unsigned char c = (unsigned char)view.data[i];
    if (c == '"' || c == '\\') {
      fputc('\\', stdout);
      fputc(c, stdout);
    } else if (c >= 0x20 && c < 0x7f) {
      fputc(c, stdout);
    } else {
      printf("\\u%04x", c);
    }
  }
}

static int fail_status(const char* label, loomc_status_t status) {
  loomc_string_view_t message = loomc_status_message(status);
  fprintf(stderr, "%s failed", label);
  if (!loomc_string_view_is_empty(message)) {
    fprintf(stderr, ": %.*s", (int)message.size, message.data);
  }
  fprintf(stderr, "\n");
  loomc_status_code_t code = loomc_status_consume_code(status);
  return code == LOOMC_STATUS_OK ? 1 : (int)code;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: %s module.loom [@symbol]\n", argv[0]);
    return 2;
  }

  const char* path = argv[1];
  const char* symbol = argc == 3 ? argv[2] : NULL;
  loomc_allocator_t allocator = loomc_allocator_system();

  loomc_target_environment_t* target_environment = NULL;
  loomc_status_t status =
      loomc_target_environment_create_amdgpu(allocator, &target_environment);
  if (!loomc_status_is_ok(status)) {
    return fail_status("loomc_target_environment_create_amdgpu", status);
  }

  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(loomc_context_target_options_t),
      .target_environment = target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(loomc_context_options_t),
      .next = &target_options,
  };

  loomc_context_t* context = NULL;
  status = loomc_context_create(&context_options, allocator, &context);
  loomc_target_environment_release(target_environment);
  if (!loomc_status_is_ok(status)) return fail_status("loomc_context_create", status);

  loomc_workspace_t* workspace = NULL;
  status = loomc_workspace_create(NULL, allocator, &workspace);
  if (!loomc_status_is_ok(status)) {
    loomc_context_release(context);
    return fail_status("loomc_workspace_create", status);
  }

  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  status = loomc_module_deserialize_from_path(
      context, workspace, loomc_make_cstring_view(path), NULL, allocator,
      &module, &result);
  if (!loomc_status_is_ok(status)) {
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    return fail_status("loomc_module_deserialize_from_path", status);
  }
  if (!loomc_result_succeeded(result) || module == NULL) {
    fprintf(stderr, "module deserialize produced diagnostics\n");
    loomc_result_release(result);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    return 1;
  }
  loomc_result_release(result);
  result = NULL;

  loomc_module_function_query_options_t query_options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      .structure_size = sizeof(loomc_module_function_query_options_t),
      .kind = LOOMC_MODULE_FUNCTION_KIND_KERNEL,
  };

  loomc_host_size_t function_count = 0;
  status = loomc_module_query_functions(module, &query_options, allocator, 0,
                                        NULL, &function_count, &result);
  if (!loomc_status_is_ok(status)) {
    loomc_module_release(module);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    return fail_status("loomc_module_query_functions", status);
  }
  if (!loomc_result_succeeded(result)) {
    fprintf(stderr, "function query produced diagnostics\n");
    loomc_result_release(result);
    loomc_module_release(module);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    return 1;
  }
  loomc_result_release(result);
  result = NULL;

  loomc_module_function_t* functions = NULL;
  if (function_count != 0) {
    status = loomc_allocator_malloc(allocator, function_count * sizeof(*functions),
                                    (void**)&functions);
    if (!loomc_status_is_ok(status)) {
      loomc_module_release(module);
      loomc_workspace_release(workspace);
      loomc_context_release(context);
      return fail_status("loomc_allocator_malloc", status);
    }
  }

  status = loomc_module_query_functions(module, &query_options, allocator,
                                        function_count, functions,
                                        &function_count, &result);
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, functions);
    loomc_module_release(module);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    return fail_status("loomc_module_query_functions", status);
  }
  if (!loomc_result_succeeded(result)) {
    fprintf(stderr, "function query produced diagnostics\n");
    loomc_result_release(result);
    loomc_allocator_free(allocator, functions);
    loomc_module_release(module);
    loomc_workspace_release(workspace);
    loomc_context_release(context);
    return 1;
  }
  loomc_result_release(result);

  printf("{\"functions\":[");
  bool first = true;
  for (loomc_host_size_t i = 0; i < function_count; ++i) {
    if (!view_equals_symbol(functions[i].symbol_name, symbol)) continue;
    loomc_module_kernel_function_info_t info;
    bool has_kernel_info =
        loomc_module_function_try_get_kernel_info(module, &functions[i], &info);
    if (!has_kernel_info) continue;

    if (!first) printf(",");
    first = false;
    bool has_count =
        (info.flags & LOOMC_MODULE_KERNEL_FUNCTION_FLAG_HAS_STATIC_DISPATCH_WORKGROUP_COUNT) != 0;
    bool has_size =
        (info.flags & LOOMC_MODULE_KERNEL_FUNCTION_FLAG_HAS_STATIC_WORKGROUP_SIZE) != 0;
    printf("{\"symbol\":\"");
    print_view_json(functions[i].symbol_name);
    printf("\",\"kind\":%d,\"has_static_dispatch_workgroup_count\":%s,",
           (int)functions[i].kind, has_count ? "true" : "false");
    printf("\"static_dispatch_workgroup_count\":[%u,%u,%u],",
           info.static_dispatch_workgroup_count.x,
           info.static_dispatch_workgroup_count.y,
           info.static_dispatch_workgroup_count.z);
    printf("\"has_static_workgroup_size\":%s,", has_size ? "true" : "false");
    printf("\"static_workgroup_size\":[%u,%u,%u]}",
           info.static_workgroup_size.x, info.static_workgroup_size.y,
           info.static_workgroup_size.z);
  }
  printf("]}\n");

  loomc_allocator_free(allocator, functions);
  loomc_module_release(module);
  loomc_workspace_release(workspace);
  loomc_context_release(context);
  return 0;
}
