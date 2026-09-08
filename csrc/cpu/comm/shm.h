// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#ifndef __SHM_COLLECTIVES__
#define __SHM_COLLECTIVES__
void shm_initialize(int size, int rank, const char* addr_string, const char* port_string);
void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size);
void barrier_wait(int root_idx, int num_ranks);
#endif
