extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (!event_value) return 0;
    *event_value = 0;
    if (!g_selected_readback_event) {
        cudaError_t err =
            cudaEventCreateWithFlags(&g_selected_readback_event,
                                     cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "selected readback event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    cudaError_t err = cudaEventRecord(g_selected_readback_event, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *event_value = ++g_selected_readback_event_value;
    return 1;
}

extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label) {
    if (event_value == 0 || !g_selected_readback_event) return 0;
    cudaError_t err = cudaEventSynchronize(g_selected_readback_event);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback wait failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    return ds4_gpu_commit_and_wait_selected_readback(event_value, label);
}

extern "C" int ds4_gpu_tensor_read_after_selected_event(
        const ds4_gpu_tensor *tensor,
        uint64_t offset,
        void *data,
        uint64_t bytes,
        uint64_t event_value,
        const char *label) {
    if (!tensor || !data || offset > tensor->bytes ||
        bytes > tensor->bytes - offset ||
        event_value == 0 ||
        !g_selected_readback_event) {
        return 0;
    }
    if (!g_selected_readback_stream) {
        cudaError_t err =
            cudaStreamCreateWithFlags(&g_selected_readback_stream,
                                      cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "selected readback stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
#ifdef __HIP_PLATFORM_AMD__
    cudaError_t err = hipStreamWaitEvent(g_selected_readback_stream,
                                         g_selected_readback_event,
                                         0);
#else
    cudaError_t err = cudaStreamWaitEvent(g_selected_readback_stream,
                                          g_selected_readback_event,
                                          0);
#endif
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback stream wait failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemcpyAsync(data,
                          (const char *)tensor->ptr + offset,
                          (size_t)bytes,
                          cudaMemcpyDeviceToHost,
                          g_selected_readback_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback copy failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaStreamSynchronize(g_selected_readback_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback sync failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    int ok = ds4_gpu_set_model_fd(fd);
    g_model_fd_host_base = model_map ? model_map : g_model_host_base;
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(
        ds4_gpu_tensor *dst,
        uint64_t dst_offset,
        const ds4_gpu_tensor *src,
        uint64_t src_offset,
        uint64_t count) {
    if (!dst || !src || !dst->ptr || !src->ptr) return 0;
    if ((dst_offset % sizeof(__half)) != 0 || (src_offset % sizeof(float)) != 0) return 0;
    if (dst_offset > dst->bytes || src_offset > src->bytes) return 0;
    if (count > (UINT64_MAX / sizeof(__half)) || count > (UINT64_MAX / sizeof(float))) return 0;
    uint64_t dst_bytes = count * sizeof(__half);
    uint64_t src_bytes = count * sizeof(float);
    if (dst_bytes > dst->bytes - dst_offset || src_bytes > src->bytes - src_offset) return 0;
    if (count == 0) return 1;
    f32_to_f16_kernel<<<(count + 255u) / 256u, 256>>>(
            (__half *)((char *)dst->ptr + dst_offset),
            (const float *)((const char *)src->ptr + src_offset),
            count);
    return cuda_ok(cudaGetLastError(), "tensor copy f32 to f16 launch");
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" int ds4_gpu_preload_q4_expert_tables(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t n_total_expert) {
    (void)model_map;
    (void)model_size;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    (void)n_total_expert;
    return 0;
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_routed_moe_selected_override_n = 0;
    g_stream_selected_cache.loaded = 0;
    g_stream_batch_selected_cache.loaded = 0;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_stream_expert_cache_budget = experts;
}

/* Fail-closed correctness floor from the adaptive planner (ds4_ssd.c).  The
 * ROCm cache does not yet shrink under live pressure the way Metal does, so
 * the floor currently guards only the startup budget; storing it keeps the
 * model contract visible to shared engine code and future enforcement. */
static uint32_t g_stream_expert_cache_required_floor;

extern "C" void ds4_gpu_set_streaming_expert_cache_required_floor(
        uint32_t experts) {
    g_stream_expert_cache_required_floor = experts;
    if (experts != 0 && g_stream_expert_cache_budget != 0 &&
        g_stream_expert_cache_budget < experts) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming expert cache budget %u is below "
                "the model's correctness floor %u\n",
                g_stream_expert_cache_budget,
                experts);
    }
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_slab_target_bytes(
        uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    (void)free_b;
    return (uint64_t)total_b;
}

/*
 * Linux/ROCm host memory snapshot for the adaptive SSD expert-cache planner.
 *
 * Field mapping from the Darwin collector (ds4_metal.m) to Linux sources:
 *   physical_bytes     <- /proc/meminfo MemTotal
 *   recommended_bytes  <- hipMemGetInfo() total: on UMA APUs the GTT pool is
 *                         the platform's working-set envelope, the closest
 *                         analogue of recommendedMaxWorkingSetSize
 *   task_footprint     <- /proc/self/status VmRSS
 *   free_bytes         <- MemFree (MemAvailable would double-count the
 *                         file-backed credit the planner adds separately)
 *   purgeable_bytes    <- 0 (no Linux analogue)
 *   inactive_bytes     <- Inactive(file): reclaimable page-cache, the
 *                         conservative counterpart of Mach inactive_count
 *   file_backed_bytes  <- Cached
 *   pressure_normal    <- PSI /proc/pressure/memory, "some avg10" < 5.0
 *
 * Motivating failure (Radeon 780M, 64 GB, GTT 48 GiB): with the legacy fixed
 * pool/8 floor the expert cache peaked after ~150-250 generated tokens and
 * the kernel TTM evicted the GPU queue buffers themselves — dmesg shows
 * "Not enough memory for command submission!" and "Freeing queue vital
 * buffer, queue evicted" — freezing generation with an idle GPU.  A
 * host-aware budget prevents it; this snapshot is what the planner needs to
 * compute one.
 */
extern "C" int ds4_gpu_host_memory_snapshot(ds4_ssd_host_memory *out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
#if !defined(__linux__)
    return 0;
#else
    size_t gtt_free = 0;
    size_t gtt_total = 0;
    if (cudaMemGetInfo(&gtt_free, &gtt_total) != cudaSuccess || gtt_total == 0) {
        (void)cudaGetLastError();
        return 0;
    }

    uint64_t mem_total = 0, mem_free = 0, inactive_file = 0, cached = 0;
    FILE *mi = fopen("/proc/meminfo", "r");
    if (!mi) return 0;
    char line[160];
    while (fgets(line, sizeof(line), mi)) {
        unsigned long long kb = 0;
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) mem_total = (uint64_t)kb << 10;
        else if (sscanf(line, "MemFree: %llu kB", &kb) == 1) mem_free = (uint64_t)kb << 10;
        else if (sscanf(line, "Inactive(file): %llu kB", &kb) == 1) inactive_file = (uint64_t)kb << 10;
        else if (sscanf(line, "Cached: %llu kB", &kb) == 1) cached = (uint64_t)kb << 10;
    }
    fclose(mi);
    if (mem_total == 0) return 0;

    uint64_t vm_rss = 0;
    FILE *st = fopen("/proc/self/status", "r");
    if (st) {
        while (fgets(line, sizeof(line), st)) {
            unsigned long long kb = 0;
            if (sscanf(line, "VmRSS: %llu kB", &kb) == 1) {
                vm_rss = (uint64_t)kb << 10;
                break;
            }
        }
        fclose(st);
    }

    out->physical_bytes = mem_total;
    out->recommended_bytes = (uint64_t)gtt_total;
    out->task_footprint_bytes = vm_rss;
    out->free_bytes = mem_free;
    out->purgeable_bytes = 0;
    out->inactive_bytes = inactive_file;
    out->file_backed_bytes = cached;

    FILE *psi = fopen("/proc/pressure/memory", "r");
    if (psi) {
        while (fgets(line, sizeof(line), psi)) {
            float avg10 = 0.0f;
            if (sscanf(line, "some avg10=%f", &avg10) == 1) {
                out->pressure_status_available = true;
                out->pressure_normal = avg10 < 5.0f;
                break;
            }
        }
        fclose(psi);
    }
    return 1;
#endif
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return g_ssd_streaming_mode ? g_stream_expert_cache_budget : 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return (uint32_t)g_stream_resident_experts.size();
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_resident_cache_release();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return ds4_gpu_stream_expert_cache_configured_count();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!table) return 0;
    if (!cuda_stream_selected_load(table->model_map,
                                   table->model_size,
                                   table->layer,
                                   selected_ids,
                                   table->n_total_expert,
                                   n_selected,
                                   table->gate_offset,
                                   table->up_offset,
                                   table->down_offset,
                                   table->gate_expert_bytes,
                                   table->down_expert_bytes)) {
        return 0;
    }
    return cuda_stream_selected_finish_pending_missing(0);
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!table) return 0;
    return cuda_stream_selected_load(table->model_map,
                                     table->model_size,
                                     table->layer,
                                     selected_ids,
                                     table->n_total_expert,
                                     n_selected,
                                     table->gate_offset,
                                     table->up_offset,
                                     table->down_offset,
                                     table->gate_expert_bytes,
                                     table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!table) return 0;
    const ds4_gpu_tensor *selected_exec = NULL;
    const char **gate_ptrs = NULL;
    const char **up_ptrs = NULL;
    const char **down_ptrs = NULL;
    uint32_t unique = 0;
    return cuda_stream_batch_selected_prepare_from_host(table->model_map,
                                                        table->model_size,
                                                        table->layer,
                                                        selected_ids,
                                                        n_tokens,
                                                        table->n_total_expert,
                                                        n_selected,
                                                        table->gate_offset,
                                                        table->up_offset,
                                                        table->down_offset,
                                                        table->gate_expert_bytes,
                                                        table->down_expert_bytes,
                                                        &selected_exec,
                                                        &gate_ptrs,
                                                        &up_ptrs,
                                                        &down_ptrs,
                                                        &unique,
                                                        1);
}

extern "C" int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table) {
    if (!table) return 0;
    return cuda_stream_layer_expert_cache_load(table->model_map,
                                               table->model_size,
                                               table->layer,
                                               table->n_total_expert,
                                               table->gate_offset,
                                               table->up_offset,
                                               table->down_offset,
                                               table->gate_expert_bytes,
                                               table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected) {
    if (!table) return 0;
    return cuda_stream_layer_expert_cache_seed_selected(table->model_map,
                                                        table->layer,
                                                        selected,
                                                        n_tokens,
                                                        n_seed_tokens,
                                                        table->n_total_expert,
                                                        n_selected,
                                                        table->gate_offset,
                                                        table->up_offset,
                                                        table->down_offset,
                                                        table->gate_expert_bytes,
                                                        table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_release_layer_cache(void) {
    cuda_stream_layer_expert_cache_release();
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    (void)table;
    (void)expert_ids;
    (void)expert_priorities;
    (void)n_experts;
    return 1;
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(
        const int32_t *selected,
        uint32_t n_selected) {
    if (n_selected > DS4_ROCM_N_EXPERT_USED || (!selected && n_selected != 0)) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_routed_moe_selected_override[i] = selected[i];
    }
    g_routed_moe_selected_override_n = n_selected;
    return 1;
}
