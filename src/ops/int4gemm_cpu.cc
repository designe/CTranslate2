#include <ctranslate2/ops/gemm.h>
#include "cpu/cpu_isa.h"
#if defined(__AVX512F__)
#  define TARGET_ISA cpu::CpuIsa::AVX512
#  include "cpu/vec_avx512.h"
#elif defined(__AVX2__)
#  define TARGET_ISA cpu::CpuIsa::AVX2
#  include "cpu/vec_avx.h"
#elif defined(__AVX__)
#  define TARGET_ISA cpu::CpuIsa::AVX
#  include "cpu/vec_avx.h"
#elif (defined(__ARM_NEON) && !defined(CT2_WITH_CPU_DISPATCH)) || defined(USE_NEON)
#  define TARGET_ISA cpu::CpuIsa::NEON
#  include "cpu/vec_neon.h"
#else
#  define TARGET_ISA cpu::CpuIsa::GENERIC
#  include "cpu/vec.h"
#  ifndef __ct2_align32__
#    define __ct2_align32__
#  endif
#endif

namespace ctranslate2 {
  namespace ops {
    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       const StorageView& scaleAndZero,
                       StorageView& c) const {
      // Get matrix dimensions
      const auto m = a.rank() == 3 ? a.dim(0) * a.dim(1) : a.dim(0);
      const auto k = a.rank() == 3 ? a.dim(2) : a.dim(1);
      const auto nTiles = b.dim(0);
      constexpr int32_t kNTileSize = 8;
      const auto n = nTiles * kNTileSize;
      const int32_t innerKTiles = b.dim(3) * 2;

      // =================================================================================
      // Part 1: Dequantization
      // 이 부분은 메모리 접근 패턴이 비연속적이라 SIMD 가속이 어려움
      // =================================================================================
      StorageView b_dequant(b.device(), scaleAndZero.dtype());
      b_dequant.resize({k, n});
      auto* b_dequant_data = b_dequant.data<bfloat16_t>();
      const auto* b_packed_data = b.data<int32_t>();
      const auto* q_data = scaleAndZero.data<bfloat16_t>();
      const auto q_n_dim = scaleAndZero.dim(1);

      constexpr int32_t kKTileSize = 16;
      const auto kSuperTiles = b.dim(1);

      const auto b_stride0 = b.stride(0);
      const auto b_stride1 = b.stride(1);
      const auto b_stride2 = b.stride(2);

      for (int32_t nTile = 0; nTile < nTiles; ++nTile) {
        for (int32_t kOuterTile = 0; kOuterTile < kSuperTiles; ++kOuterTile) {
          for (int32_t t = 0; t < 32; ++t) {
            for (int32_t inner_k_half = 0; inner_k_half < innerKTiles / 2; ++inner_k_half) {
              const size_t b_idx = nTile * b_stride0 + kOuterTile * b_stride1 + t * b_stride2 + inner_k_half;
              const int32_t pack = b_packed_data[b_idx];

              uint32_t v[8];
              v[0] = (pack >> 0) & 0xF; v[2] = (pack >> 4) & 0xF; v[4] = (pack >> 8) & 0xF;
              v[6] = (pack >> 12) & 0xF; v[1] = (pack >> 16) & 0xF; v[3] = (pack >> 20) & 0xF;
              v[5] = (pack >> 24) & 0xF; v[7] = (pack >> 28) & 0xF;

              const int32_t n_coord = nTile * kNTileSize + (t / 4);
              if (n_coord >= n) continue;

              const int32_t innerKTile = inner_k_half * 2;
              const int32_t k_base_lane = (t % 4) * 2;
              int32_t ks[8];

              const auto kBase0 = (kOuterTile * innerKTiles + innerKTile) * kKTileSize;
              ks[0] = kBase0 + k_base_lane; ks[1] = ks[0] + 1; ks[2] = ks[0] + 8; ks[3] = ks[2] + 1;
              const auto kBase1 = kBase0 + kKTileSize;
              ks[4] = kBase1 + k_base_lane; ks[5] = ks[4] + 1; ks[6] = ks[4] + 8; ks[7] = ks[6] + 1;

              for (int i = 0; i < 8; ++i) {
                const int32_t k_coord = ks[i];
                if (k_coord >= k) continue;
                const int8_t s_v = (v[i] & 8) ? (v[i] - 16) : v[i];

                const int32_t q_group_idx = k_coord / _group_size;
                const bfloat16_t scale = q_data[(q_group_idx * q_n_dim + n_coord) * 2 + 0];
                const bfloat16_t zero = q_data[(q_group_idx * q_n_dim + n_coord) * 2 + 1];

                const float dequant_val = static_cast<float>(s_v) * static_cast<float>(scale) + static_cast<float>(zero);
                b_dequant_data[k_coord * n + n_coord] = static_cast<bfloat16_t>(dequant_val);
              }
            }
          }
        }
      }

      // =================================================================================
      // Part 2: SIMD-accelerated GEMM (C = A * B_dequant_transposed)
      // =================================================================================
      c.resize({m, n});
      const auto* a_data = a.data<bfloat16_t>();
      auto* c_data = c.data<bfloat16_t>();

      // b_dequant_data는 float으로 변환하여 임시 저장
      StorageView b_dequant_f(b.device(), DataType::FLOAT32);
      b_dequant_f.resize({k, n});
      auto* b_dequant_data_f = b_dequant_f.data<float>();
      for(dim_t i = 0; i < b_dequant.size(); ++i) {
        b_dequant_data_f[i] = static_cast<float>(b_dequant_data[i]);
      }

      using V = cpu::Vec<float, TARGET_ISA>;
      const dim_t vec_width = V::width;

      for (int32_t i = 0; i < m; ++i) {
        int32_t j = 0;

        // 주된 Vectorized 루프
        for (; j + vec_width <= n; j += vec_width) {
          V::value_type c_vec = V::load(0.0f); // 8개의 합계를 저장할 벡터 레지스터 초기화
          for (int32_t l = 0; l < k; ++l) {
            V::value_type a_vec = V::load(static_cast<float>(a_data[i * k + l])); // A의 스칼라 값을 8개 복사
            V::value_type b_vec = V::load(b_dequant_data_f + l * n + j); // B의 연속된 8개 값을 로드
            c_vec = V::mul_add(a_vec, b_vec, c_vec); // Fused Multiply-Add: c_vec += a_vec * b_vec
          }
          // bfloat16_t 타입으로 변환하여 C 행렬에 8개의 결과값을 저장
          __ct2_align32__ float tmp[vec_width];
          V::store(c_vec, tmp);
          for(dim_t v = 0; v < vec_width; ++v) {
            c_data[i * n + j + v] = static_cast<bfloat16_t>(tmp[v]);
          }
        }

        // `n`이 8의 배수가 아닐 경우 나머지 부분을 처리
        if (j < n) {
          const dim_t count = n - j;
          V::value_type c_vec = V::load(0.0f);
          for (int32_t l = 0; l < k; ++l) {
            V::value_type a_vec = V::load(static_cast<float>(a_data[i * k + l]));
            V::value_type b_vec = V::load(b_dequant_data_f + l * n + j, count, 0.0f); // 마스크 로드
            c_vec = V::mul_add(a_vec, b_vec, c_vec);
          }
          __ct2_align32__ float tmp[vec_width];
          V::store(c_vec, tmp);
          for(dim_t v = 0; v < count; ++v) {
            c_data[i * n + j + v] = static_cast<bfloat16_t>(tmp[v]);
          }
        }
      }
    }

    template <>
    void Gemm::convert_weight_to_int4pack<Device::CPU>(const StorageView& a,
                                                       StorageView& b,
                                                       int32_t innerKTiles) {
      const auto n = a.dim(0);
      const auto k = a.dim(1);
      const auto* in_data = a.data<int32_t>();

      constexpr int32_t kNTileSize = 8;
      constexpr int32_t kKTileSize = 16;
      const auto nTiles = (n + kNTileSize - 1) / kNTileSize;
      
      // k-tiles are packed back to back in the innermost dimension in order to
      // allow for 4/8/16 byte loads
      // kSuperTiles is the number of k-tiles assuming k is innerKTiles * kKTileSize
      const auto kSuperTiles = (k + (innerKTiles * kKTileSize) - 1) / (innerKTiles * kKTileSize);
      
      // each block handles `innerKTiles` k-tiles.
      // 2 k-tiles are a single int32    
      b.resize({nTiles, kSuperTiles, 32, innerKTiles / 2});
      auto* out_data = b.data<int32_t>();

      const auto b_stride0 = b.stride(0);
      const auto b_stride1 = b.stride(1);
      const auto b_stride2 = b.stride(2);

      for (int32_t nTile = 0; nTile < nTiles; ++nTile) {
        for (int32_t kOuterTile = 0; kOuterTile < kSuperTiles; ++kOuterTile) {
          for (int32_t t = 0; t < 32; ++t) {
            for (int32_t inner_k_half = 0; inner_k_half < innerKTiles / 2; ++inner_k_half) {
              const int32_t innerKTile = inner_k_half * 2;
              const int32_t n0 = nTile * kNTileSize + (t / 4);
              uint32_t v[8] = {};

              if (n0 < n) {
                const int32_t k_base_lane = (t % 4) * 2;
                int32_t ks[8];

                const auto kBase0 = (kOuterTile * innerKTiles + innerKTile) * kKTileSize;
                ks[0] = kBase0 + k_base_lane; ks[1] = ks[0] + 1; ks[2] = ks[0] + 8; ks[3] = ks[2] + 1;
                const auto kBase1 = kBase0 + kKTileSize;
                ks[4] = kBase1 + k_base_lane; ks[5] = ks[4] + 1; ks[6] = ks[4] + 8; ks[7] = ks[6] + 1;

                for (int i = 0; i < 8; ++i) {
                  if (ks[i] < k) {
                    v[i] = in_data[n0 * k + ks[i]] & 0xF;
                  }
                }
              }

              const int32_t pack = (v[7] << 28) | (v[5] << 24) | (v[3] << 20) | (v[1] << 16) |
                                   (v[6] << 12) | (v[4] << 8) | (v[2] << 4) | v[0];
              
              const size_t out_idx = nTile * b_stride0 + kOuterTile * b_stride1 + t * b_stride2 + inner_k_half;
              out_data[out_idx] = pack;
            }
          }
        }
      }
    }


#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Gemm::compute<Device::CPU, int32_t, T>(const StorageView& a,         \
                                          const StorageView& b,         \
                                          const StorageView& scaleAndZero, \
                                          StorageView& c) const;

    DECLARE_IMPL(bfloat16_t)
  }
}