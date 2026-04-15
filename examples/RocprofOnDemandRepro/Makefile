TRACY_PUBLIC := ../../public
HIPCC        := /opt/rocm/bin/hipcc
CXX          := g++
ROCM_LIB     := /opt/rocm/lib

TRACY_SRCS   := $(TRACY_PUBLIC)/TracyClient.cpp
INCLUDES     := -I$(TRACY_PUBLIC)
LIBS         := -L$(ROCM_LIB) -lrocprofiler-sdk -lpthread -ldl

# On-demand mode (the default for this repro) — profiling starts when
# a client connects, not at program launch.
DEFINES      := -DTRACY_ENABLE -DTRACY_ON_DEMAND -DTRACY_ROCPROF
CXXFLAGS     := -O2 $(DEFINES)
HIPCCFLAGS   := -O2 $(DEFINES)

.PHONY: all clean

all: repro

repro: repro.cpp tracy_client.o
	$(HIPCC) $(HIPCCFLAGS) $(INCLUDES) -o $@ $< tracy_client.o $(LIBS)

tracy_client.o: $(TRACY_SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f repro tracy_client.o
