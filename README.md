# Parallel Watershed Image Segmentation

**C++ image segmentation with OpenMP parallelization, graph-based watershed processing, and performance profiling.**

This project explores how parallel computing can accelerate a marker-based watershed segmentation pipeline. Developed for my master's thesis at Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), it combines image preprocessing, distance transformation, seed detection, and flooding-based label propagation with an analysis of computational bottlenecks.

## Overview

Watershed segmentation treats an image as a topographic surface. Starting from markers, regions expand through flooding until they meet other regions, producing a partition of the image.

The project investigates two practical challenges:

- **Segmentation quality:** how preprocessing, marker selection, and region merging influence over-segmentation.
- **Computational performance:** how parallel execution and data structures affect runtime and scalability.

The implementation uses C++ for the core algorithm and OpenCV image containers and utilities. OpenMP provides CPU parallelization. Selected OpenACC kernels explore accelerator execution; complete GPU acceleration remains unfinished in the thesis.

## Main Contributions

- Implemented a segmentation pipeline covering preprocessing, Canny edge detection, city-block distance transformation, seed selection, and watershed flooding.
- Used a level-component graph to organize watershed processing and label propagation.
- Applied OpenMP directives to parallelizable stages and compared execution times with a sequential baseline.
- Investigated region merging through region adjacency graphs and nearest-neighbor graphs to address over-segmentation.
- Profiled the implementation with Intel Advisor, examining priority-queue operations, dynamic allocation, and memory access.
- Explored OpenACC data management and parallel loops for selected image-processing operations.

## Processing Pipeline

| Stage | Purpose |
| --- | --- |
| Image preparation | Load image data and prepare grayscale representations. |
| Preprocessing and edge detection | Reduce noise and identify boundaries using preprocessing operations and Canny edge detection. |
| Distance transformation | Compute city-block distances to boundaries to support marker selection. |
| Seed detection | Identify markers used to initialize region labels. |
| Watershed processing | Propagate labels through flooding and a level-component graph representation. |
| Region merging | Reduce excessive partitioning by considering similarities between adjacent regions. |
| Performance evaluation | Compare sequential and parallel runtimes and inspect scaling behavior. |

## Technology

| Technology | Role |
| --- | --- |
| C++ | Core algorithm and data structures |
| OpenCV | Image representation and image-processing utilities |
| OpenMP | CPU parallel execution |
| OpenACC | Exploratory accelerator directives and data transfers |
| Intel Advisor | Performance profiling and bottleneck analysis |
| Linux | Development and benchmark environment |

## Benchmark Results

The thesis reports the following sequential and OpenMP timings for a landscape image containing **612 × 262 pixels**. These are historical results from the thesis, not newly reproduced measurements.

| Reported measurement | Sequential | Parallel | Calculated speedup | Runtime reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 80 ms | 39 ms | 2.05× | 51.3% |
| 2 | 79 ms | 34 ms | 2.32× | 57.0% |
| 3 | 88 ms | 42 ms | 2.10× | 52.3% |

Speedup is calculated as `sequential time / parallel time`. Runtime reduction is calculated as `1 - parallel time / sequential time`.

### Scaling Study

| Reported parallel worker count | Runtime |
| ---: | ---: |
| 2 | 53.0 ms |
| 4 | 41.0 ms |
| 6 | 37.5 ms |
| 8 | 32.6 ms |
| 10 | 32.3 ms |
| 12 | 31.6 ms |

The reported machine used an **Intel Xeon W-1390P at 3.50 GHz**, with **8 physical cores and 16 logical CPUs**, on Ubuntu Linux. The thesis labels its parallel configurations as cores or processors; counts above eight should not be interpreted as physical-core counts on this machine.

The measurements show diminishing returns at higher worker counts. The thesis attributes this behavior to memory-related limitations and remaining sequential work.

*Source: thesis Section 6.2, Figures 58–60. Results depend on the image, compiler, configuration, and hardware; they do not establish a general speedup across datasets.*

## Performance Findings

Profiling of the sequential implementation identified watershed processing as approximately **72.14%** of total runtime, with priority-queue operations accounting for almost **40%** of total runtime.

The analysis highlights several optimization targets:

- Allocation and deallocation of dynamically stored pixel objects.
- Insertion, removal, and ordering operations in the priority queue.
- Data movement and repeated access to image containers.
- Dependencies in label propagation that constrain parallel execution.

These findings motivate improvements to memory layout, allocation patterns, and work distribution rather than relying solely on additional threads.

## Limitations and Future Work

- **Parameter sensitivity:** Canny and distance-transform thresholds depend on image texture and influence marker quality.
- **Over-segmentation:** marker placement and flat image regions can produce excessive regions.
- **Plateau handling:** improved handling of equal-valued regions remains an optimization opportunity.
- **GPU implementation:** the OpenACC work is exploratory; a complete, validated GPU pipeline and comprehensive GPU benchmarks remain future work.
- **Evaluation scope:** the reported runtime measurements cover a limited experimental setting. Broader datasets, segmentation-quality metrics, and reproducible benchmark configurations would strengthen the evaluation.

## Build and Reproducibility

The technologies above describe the thesis implementation. Exact dependency versions, build targets, executable names, and input/output arguments must be documented from the corresponding source code before a reproducible quick-start command can be provided.

This README summarizes the thesis; it does not imply that source code, test images, or build scripts are included in the repository.

https://www10.cs.fau.de/publications/theses/2023/Master_ZhangLiang.pdf
