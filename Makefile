CC ?= cc
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
# Qwen's stable softmax rejects non-finite logits; retain that branch while
# keeping the remaining fast-math optimizations used by the scalar CPU path.
QWEN_CFLAGS ?= -fno-finite-math-only
DEPFLAGS ?= -MMD -MP

BUILD_GIT_SHA ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
BUILD_GIT_SUFFIX ?= $(shell test -z "$$(git status --porcelain --untracked-files=normal 2>/dev/null)" || printf '%s' -dirty)
# Pass the sha as a bare token and stringify it in ds4_build.c: a quoted
# -D value does not survive the extra shell evaluation in recursive targets
# (strix-halo/cuda pass CFLAGS="$(CFLAGS) ..." through $(MAKE)).
CFLAGS += -DDS4_BUILD_GIT_SHA=$(BUILD_GIT_SHA)$(BUILD_GIT_SUFFIX)

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)

BUILD_ROOT ?= build
PROGRAMS := ds4 ds4-server ds4-bench ds4-eval ds4-agent

.PHONY: all help clean test model-free-test cpu cuda cuda-spark cuda-generic cuda-regression FORCE \
	strix-halo rocm metal build-isolation-test q4k-dot-test qwen-metadata-test \
	qwen-reference-test qwen-unicode-test qwen-tokenizer-test \
	$(PROGRAMS) ds4_test ds4_agent_test

ifeq ($(UNAME_S),Darwin)

# A build profile owns every object and binary it produces.  In particular, a
# CPU build can never satisfy a Metal prerequisite (or replace a Metal binary).
# The same profile convention is available to future CUDA/ROCm layouts:
#     build/<backend>-<backend-arch>-<host-arch>/{obj,bin}
METAL_PROFILE := metal-$(UNAME_M)
CPU_PROFILE := cpu-$(UNAME_M)
METAL_OBJDIR := $(BUILD_ROOT)/$(METAL_PROFILE)/obj
METAL_BINDIR := $(BUILD_ROOT)/$(METAL_PROFILE)/bin
CPU_OBJDIR := $(BUILD_ROOT)/$(CPU_PROFILE)/obj
CPU_BINDIR := $(BUILD_ROOT)/$(CPU_PROFILE)/bin

METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal

METAL_CORE_OBJS := $(addprefix $(METAL_OBJDIR)/,ds4.o ds4_build.o ds4_distributed.o ds4_ssd.o ds4_qwen.o ds4_qwen_unicode.o ds4_metal.o)
CPU_CORE_OBJS := $(addprefix $(CPU_OBJDIR)/,ds4.o ds4_build.o ds4_distributed.o ds4_ssd.o ds4_qwen.o ds4_qwen_unicode.o)

METAL_BINS := $(addprefix $(METAL_BINDIR)/,$(PROGRAMS))
CPU_BINS := $(addprefix $(CPU_BINDIR)/,$(PROGRAMS))

METAL_TEST_BINS := \
	$(METAL_BINDIR)/ds4_test \
	$(METAL_BINDIR)/ds4_agent_test \
	$(METAL_BINDIR)/test_q4k_dot \
	$(METAL_BINDIR)/test_q4k_top8 \
	$(METAL_BINDIR)/test_qwen_session \
	$(METAL_BINDIR)/test_qwen_gdn_ref \
	$(METAL_BINDIR)/test_qwen_attention_ref \
	$(METAL_BINDIR)/test_qwen_state \
	$(METAL_BINDIR)/test_qwen_unicode \
	$(METAL_BINDIR)/test_qwen_tokenizer \
	$(METAL_BINDIR)/test_ssd_residency

all: metal

help:
	@echo "DS4 build targets:"
	@echo "  make / make metal Build Metal and publish ./ds4* -> $(METAL_BINDIR)"
	@echo "  make cpu          Build CPU-only binaries in $(CPU_BINDIR) (never changes ./ds4*)"
	@echo "  make test         Build and run the Metal test suite"
	@echo "  make model-free-test"
	@echo "                    Run all Metal gates that do not require a GGUF"
	@echo "  make build-isolation-test"
	@echo "                    Prove Metal -> CPU -> Metal cannot mix artifacts"
	@echo "  make clean        Remove build outputs and published root binaries"

# Root binaries are a Metal-only compatibility surface on macOS.  These targets
# are phony so an old regular CPU binary is replaced even when its timestamp is
# newer than the namespaced Metal binary.
metal: $(PROGRAMS)

$(PROGRAMS): %: $(METAL_BINDIR)/%
	@rm -f "$@"
	@ln -s "$<" "$@"

cpu: $(CPU_BINS)
	@echo "CPU-only binaries: $(CPU_BINDIR)"

$(METAL_BINDIR)/ds4: \
	$(METAL_OBJDIR)/ds4_cli.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/linenoise.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/ds4-server: \
	$(METAL_OBJDIR)/ds4_server.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_kvstore.o $(METAL_OBJDIR)/rax.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/ds4-bench: \
	$(METAL_OBJDIR)/ds4_bench.o $(METAL_OBJDIR)/ds4_help.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/ds4-eval: \
	$(METAL_OBJDIR)/ds4_eval.o $(METAL_OBJDIR)/ds4_help.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/ds4-agent: \
	$(METAL_OBJDIR)/ds4_agent.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_web.o $(METAL_OBJDIR)/ds4_kvstore.o \
	$(METAL_OBJDIR)/linenoise.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(CPU_BINDIR)/ds4: \
	$(CPU_OBJDIR)/ds4_cli.o $(CPU_OBJDIR)/ds4_help.o \
	$(CPU_OBJDIR)/linenoise.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/ds4-server: \
	$(CPU_OBJDIR)/ds4_server.o $(CPU_OBJDIR)/ds4_help.o \
	$(CPU_OBJDIR)/ds4_kvstore.o $(CPU_OBJDIR)/rax.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/ds4-bench: \
	$(CPU_OBJDIR)/ds4_bench.o $(CPU_OBJDIR)/ds4_help.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/ds4-eval: \
	$(CPU_OBJDIR)/ds4_eval.o $(CPU_OBJDIR)/ds4_help.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/ds4-agent: \
	$(CPU_OBJDIR)/ds4_agent.o $(CPU_OBJDIR)/ds4_help.o \
	$(CPU_OBJDIR)/ds4_web.o $(CPU_OBJDIR)/ds4_kvstore.o \
	$(CPU_OBJDIR)/linenoise.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_OBJDIR)/%.o: %.c
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(CPU_OBJDIR)/%.o: %.c
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

# Build provenance is intentionally refreshed on every invocation.  Keeping it
# in this tiny object prevents a clean/dirty transition from forcing the giant
# engine translation unit to rebuild while ensuring --build-info is truthful.
$(METAL_OBJDIR)/ds4_build.o: ds4_build.c ds4.h FORCE
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(CPU_OBJDIR)/ds4_build.o: ds4_build.c ds4.h FORCE
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/ds4_qwen.o: ds4_qwen.c ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(CPU_OBJDIR)/ds4_qwen.o: ds4_qwen.c ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/ds4_qwen_unicode.o: ds4_qwen_unicode.c ds4_qwen_unicode.h \
		ds4_qwen_unicode_data.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(CPU_OBJDIR)/ds4_qwen_unicode.o: ds4_qwen_unicode.c ds4_qwen_unicode.h \
		ds4_qwen_unicode_data.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	@mkdir -p "$(@D)"
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -c -o $@ ds4_metal.m

$(METAL_OBJDIR)/ds4_test.o: tests/ds4_test.c
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function -c -o $@ $<

$(METAL_OBJDIR)/ds4_agent_test.o: tests/ds4_agent_test.c
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function -c -o $@ $<

$(METAL_OBJDIR)/test_q4k_dot.o: tests/test_q4k_dot.c
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/test_q4k_top8.o: tests/test_q4k_top8.c ds4.c ds4.h \
		ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_session.o: tests/test_qwen_session.c ds4.c ds4.h \
		ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_tokenizer.o: tests/test_qwen_tokenizer.c ds4.c \
		ds4.h ds4_kvstore.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h tests/qwen/qwen36_tokenizer_fixture.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -c -o $@ $<

$(METAL_OBJDIR)/test_ssd_residency.o: tests/test_ssd_residency.c \
		ds4_ssd.h ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_gdn_ref.o: tests/test_qwen_gdn_ref.c ds4_qwen_ref.h \
		ds4_qwen.h tests/qwen/qwen36_gdn_golden.inc
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_attention_ref.o: tests/test_qwen_attention_ref.c \
		ds4_qwen_ref.h ds4_qwen.h tests/qwen/qwen36_attention_golden.inc
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_state.o: tests/test_qwen_state.c ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_unicode.o: tests/test_qwen_unicode.c \
		ds4_qwen_unicode.h
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/ds4_qwen_ref.o: ds4_qwen_ref.c ds4_qwen_ref.h
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -c -o $@ $<

$(METAL_BINDIR)/ds4_test: \
	$(METAL_OBJDIR)/ds4_test.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_kvstore.o $(METAL_OBJDIR)/rax.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/ds4_agent_test: \
	$(METAL_OBJDIR)/ds4_agent_test.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_web.o $(METAL_OBJDIR)/ds4_kvstore.o \
	$(METAL_OBJDIR)/linenoise.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/test_q4k_dot: $(METAL_OBJDIR)/test_q4k_dot.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm -pthread

$(METAL_BINDIR)/test_q4k_top8: \
		$(METAL_OBJDIR)/test_q4k_top8.o $(METAL_OBJDIR)/ds4_build.o \
		$(METAL_OBJDIR)/ds4_distributed.o $(METAL_OBJDIR)/ds4_ssd.o \
		$(METAL_OBJDIR)/ds4_qwen.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_qwen_session: \
		$(METAL_OBJDIR)/test_qwen_session.o $(METAL_OBJDIR)/ds4_build.o \
		$(METAL_OBJDIR)/ds4_distributed.o $(METAL_OBJDIR)/ds4_ssd.o \
		$(METAL_OBJDIR)/ds4_qwen.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_qwen_tokenizer: \
		$(METAL_OBJDIR)/test_qwen_tokenizer.o $(METAL_OBJDIR)/ds4_kvstore.o \
		$(METAL_OBJDIR)/ds4_build.o \
		$(METAL_OBJDIR)/ds4_distributed.o $(METAL_OBJDIR)/ds4_ssd.o \
		$(METAL_OBJDIR)/ds4_qwen.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_ssd_residency: \
	$(METAL_OBJDIR)/test_ssd_residency.o $(METAL_OBJDIR)/ds4_ssd.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_qwen_gdn_ref: \
		$(METAL_OBJDIR)/test_qwen_gdn_ref.o $(METAL_OBJDIR)/ds4_qwen_ref.o \
		$(METAL_OBJDIR)/ds4_qwen.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm

$(METAL_BINDIR)/test_qwen_attention_ref: \
		$(METAL_OBJDIR)/test_qwen_attention_ref.o $(METAL_OBJDIR)/ds4_qwen_ref.o \
		$(METAL_OBJDIR)/ds4_qwen.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm

$(METAL_BINDIR)/test_qwen_state: \
	$(METAL_OBJDIR)/test_qwen_state.o $(METAL_OBJDIR)/ds4_qwen.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm

$(METAL_BINDIR)/test_qwen_unicode: \
	$(METAL_OBJDIR)/test_qwen_unicode.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^

# Preserve the documented direct test-runner commands without letting a CPU
# target publish over them.
ds4_test: $(METAL_BINDIR)/ds4_test
	@rm -f "$@"
	@ln -s "$<" "$@"

ds4_agent_test: $(METAL_BINDIR)/ds4_agent_test
	@rm -f "$@"
	@ln -s "$<" "$@"

q4k-dot-test: $(METAL_BINDIR)/test_q4k_dot
	$<

qwen-metadata-test: $(METAL_BINDIR)/ds4 tests/test_qwen_metadata.py
	python3 tests/test_qwen_metadata.py $(METAL_BINDIR)/ds4

qwen-reference-test: $(METAL_BINDIR)/test_qwen_gdn_ref \
		$(METAL_BINDIR)/test_qwen_attention_ref \
		$(METAL_BINDIR)/test_qwen_unicode
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	$(METAL_BINDIR)/test_qwen_gdn_ref
	$(METAL_BINDIR)/test_qwen_attention_ref
	$(METAL_BINDIR)/test_qwen_unicode

qwen-unicode-test: $(METAL_BINDIR)/test_qwen_unicode
	python3 tests/gen_qwen_unicode.py --check
	$(METAL_BINDIR)/test_qwen_unicode

qwen-tokenizer-test: $(METAL_BINDIR)/test_qwen_tokenizer
	$(METAL_BINDIR)/test_qwen_tokenizer

model-free-test: metal ds4_test ds4_agent_test $(METAL_BINDIR)/test_q4k_dot \
		$(METAL_BINDIR)/test_q4k_top8 \
		$(METAL_BINDIR)/test_qwen_session \
		$(METAL_BINDIR)/test_qwen_gdn_ref \
		$(METAL_BINDIR)/test_qwen_attention_ref \
		$(METAL_BINDIR)/test_qwen_state \
		$(METAL_BINDIR)/test_qwen_unicode \
		$(METAL_BINDIR)/test_qwen_tokenizer \
		$(METAL_BINDIR)/test_ssd_residency
	$(METAL_BINDIR)/ds4-eval --self-test-extractors
	$(METAL_BINDIR)/ds4_agent_test
	$(METAL_BINDIR)/ds4_test --server
	$(METAL_BINDIR)/test_q4k_dot
	$(METAL_BINDIR)/test_q4k_top8
	$(METAL_BINDIR)/test_qwen_session
	$(METAL_BINDIR)/test_qwen_gdn_ref
	$(METAL_BINDIR)/test_qwen_attention_ref
	$(METAL_BINDIR)/test_qwen_state
	$(METAL_BINDIR)/test_qwen_unicode
	$(METAL_BINDIR)/test_qwen_tokenizer
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	$(METAL_BINDIR)/test_ssd_residency
	python3 tests/test_qwen_metadata.py $(METAL_BINDIR)/ds4

test: model-free-test
	$(METAL_BINDIR)/ds4_test

build-isolation-test: tests/test_build_isolation.sh
	MAKE="$(MAKE)" sh tests/test_build_isolation.sh

cuda-regression:
	@echo "cuda-regression requires a CUDA build"

-include $(wildcard $(METAL_OBJDIR)/*.d $(CPU_OBJDIR)/*.d)

else

CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CORE_OBJS = ds4.o ds4_build.o ds4_distributed.o ds4_ssd.o ds4_qwen.o ds4_qwen_unicode.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_build_cpu.o ds4_distributed.o ds4_ssd.o ds4_qwen.o ds4_qwen_unicode.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)

all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make cpu                 Build CPU-only ./ds4* binaries"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_build.o ds4_distributed.o ds4_ssd.o ds4_qwen.o ds4_qwen_unicode.o ds4_rocm.o" \
		CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h ds4_streaming_hotlist.inc
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_build.o: ds4_build.c ds4.h FORCE
	$(CC) $(CFLAGS) -c -o $@ ds4_build.c

ds4_build_cpu.o: ds4_build.c ds4.h FORCE
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_build.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_qwen.o: ds4_qwen.c ds4_qwen.h
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -c -o $@ ds4_qwen.c

ds4_qwen_unicode.o: ds4_qwen_unicode.c ds4_qwen_unicode.h \
		ds4_qwen_unicode_data.inc
	$(CC) $(CFLAGS) -c -o $@ ds4_qwen_unicode.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/test_ssd_residency: tests/test_ssd_residency.c ds4_ssd.o
	$(CC) $(CFLAGS) -I. -o $@ $^ -lm -pthread

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h ds4_streaming_hotlist.inc
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

model-free-test: ds4 ds4_test ds4_agent_test ds4-eval q4k-dot-test \
		tests/test_q4k_top8 \
		tests/test_qwen_session \
		tests/test_qwen_tokenizer \
		tests/test_qwen_gdn_ref tests/test_qwen_attention_ref \
		tests/test_qwen_state tests/test_qwen_unicode \
		tests/test_ssd_residency
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test --server
	./tests/test_q4k_top8
	./tests/test_qwen_session
	./tests/test_qwen_tokenizer
	./tests/test_qwen_gdn_ref
	./tests/test_qwen_attention_ref
	./tests/test_qwen_state
	./tests/test_qwen_unicode
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	./tests/test_ssd_residency
	python3 tests/test_qwen_metadata.py ./ds4

test: model-free-test
	./ds4_test

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

tests/test_q4k_top8: tests/test_q4k_top8.c ds4.c ds4.h ds4_ssd.h \
		ds4_distributed.h ds4_gpu.h ds4_qwen.h ds4_qwen_unicode.h \
		ds4_build.c ds4_distributed.c ds4_ssd.c ds4_qwen.c \
		ds4_qwen_unicode.c ds4_qwen_unicode_data.inc \
		ds4_streaming_hotlist.inc
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -o $@ \
		tests/test_q4k_top8.c ds4_build.c ds4_distributed.c ds4_ssd.c \
		ds4_qwen.c ds4_qwen_unicode.c $(LDLIBS)

tests/test_qwen_session: tests/test_qwen_session.c ds4.c ds4.h ds4_ssd.h \
		ds4_distributed.h ds4_gpu.h ds4_qwen.h ds4_qwen_unicode.h \
		ds4_build.c ds4_distributed.c ds4_ssd.c ds4_qwen.c \
		ds4_qwen_unicode.c ds4_qwen_unicode_data.inc \
		ds4_streaming_hotlist.inc
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -o $@ \
		tests/test_qwen_session.c ds4_build.c ds4_distributed.c ds4_ssd.c \
		ds4_qwen.c ds4_qwen_unicode.c $(LDLIBS)

tests/test_qwen_tokenizer: tests/test_qwen_tokenizer.c ds4.c ds4.h \
		ds4_kvstore.c ds4_kvstore.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h ds4_build.c ds4_distributed.c ds4_ssd.c \
		ds4_qwen.c ds4_qwen_unicode.c ds4_qwen_unicode_data.inc \
		ds4_streaming_hotlist.inc tests/qwen/qwen36_tokenizer_fixture.inc
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -o $@ \
		tests/test_qwen_tokenizer.c ds4_kvstore.c ds4_build.c ds4_distributed.c ds4_ssd.c \
		ds4_qwen.c ds4_qwen_unicode.c $(LDLIBS)

qwen-metadata-test: ds4 tests/test_qwen_metadata.py
	python3 tests/test_qwen_metadata.py ./ds4

tests/test_qwen_gdn_ref: tests/test_qwen_gdn_ref.c ds4_qwen_ref.c ds4_qwen.c \
		ds4_qwen_ref.h ds4_qwen.h tests/qwen/qwen36_gdn_golden.inc
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_gdn_ref.c ds4_qwen_ref.c ds4_qwen.c -lm

tests/test_qwen_attention_ref: tests/test_qwen_attention_ref.c ds4_qwen_ref.c \
		ds4_qwen.c ds4_qwen_ref.h ds4_qwen.h \
		tests/qwen/qwen36_attention_golden.inc
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_attention_ref.c ds4_qwen_ref.c ds4_qwen.c -lm

tests/test_qwen_state: tests/test_qwen_state.c ds4_qwen.c ds4_qwen.h
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_state.c ds4_qwen.c -lm

tests/test_qwen_unicode: tests/test_qwen_unicode.c ds4_qwen_unicode.c \
		ds4_qwen_unicode.h ds4_qwen_unicode_data.inc
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_unicode.c ds4_qwen_unicode.c

qwen-reference-test: tests/test_qwen_gdn_ref tests/test_qwen_attention_ref \
		tests/test_qwen_unicode
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	./tests/test_qwen_gdn_ref
	./tests/test_qwen_attention_ref
	./tests/test_qwen_unicode

qwen-unicode-test: tests/test_qwen_unicode
	python3 tests/gen_qwen_unicode.py --check
	./tests/test_qwen_unicode

qwen-tokenizer-test: tests/test_qwen_tokenizer
	./tests/test_qwen_tokenizer

endif

clean:
	rm -rf "$(BUILD_ROOT)"
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native \
		ds4_server_test ds4_test ds4_agent_test tests/test_q4k_dot \
		tests/test_q4k_top8 \
		tests/test_qwen_session \
		tests/test_qwen_tokenizer \
		tests/test_qwen_gdn_ref tests/test_qwen_attention_ref \
		tests/test_qwen_state tests/test_qwen_unicode \
		tests/test_ssd_residency tests/cuda_long_context_smoke \
		tests/cuda_long_context_smoke.o *.o
