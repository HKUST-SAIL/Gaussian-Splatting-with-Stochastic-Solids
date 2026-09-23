/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define MEDIAN_DEPTH_INIT

#if (defined(MEDIAN_DEPTH_INIT) + defined(MEAN_DEPTH_INIT) + defined(MOST_VISIBLE_INIT)) != 1
#error "Must define exactly one of MEDIAN_DEPTH_INIT / MEAN_DEPTH_INIT / MOST_VISIBLE_INIT"
#endif

constexpr int NUM_CHANNELS        = 3;
constexpr int BLOCK_X             = 16;
constexpr int BLOCK_Y             = 16;
constexpr int BLOCK_SIZE          = BLOCK_X * BLOCK_Y;
constexpr int SAMPLE_BATCH_SIZE   = 1;
constexpr float NEAR_PLANE        = 0.2f;
constexpr float FAR_PLANE         = 100.f;
constexpr float NORMALIZE_EPS     = 1.0E-12F;
constexpr float MIN_TRANSMITTANCE = 0.45f;
constexpr int SPLIT               = 8;

constexpr float SAMPLE_RANGE   = 0.4f;
constexpr int SPLIT_ITERATIONS = 5;

constexpr float SAMPLE_RANGE_TESTING   = 100.f;
constexpr int SPLIT_ITERATIONS_TESTING = 8;



#endif