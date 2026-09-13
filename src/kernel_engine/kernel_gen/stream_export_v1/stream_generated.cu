// Warp-generated frozen stream forward; header license: ../export/THIRD_PARTY.
#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#include "builtin.h"
#include "deterministic.h"

#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define WP_MAXNREG(n) __maxnreg__(n)
#else
#define WP_MAXNREG(n)
#endif

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13) && !defined(_DEBUG)
#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \"enable_smem_spilling\";");
#else
#define WP_ENABLE_SMEM_SPILLING()
#endif


#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)



extern "C" __global__ void stream_3bfa5cbb_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_fpost,
    wp::array_t<wp::float32> var_f1,
    wp::array_t<wp::int32> var_solid,
    wp::array_t<wp::float32> var_cx,
    wp::array_t<wp::float32> var_cy,
    wp::array_t<wp::int32> var_opp,
    wp::int32 var_nx,
    wp::int32 var_ny)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        wp::tile_shared_storage_t::init();

        wp::int32 var_0;
        wp::int32 var_1;
        const wp::int32 var_2 = 0;
        wp::float32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        bool var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 0;
        bool var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        bool var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        const wp::int32 var_27 = 1;
        bool var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::float32* var_31;
        wp::int32 var_32;
        wp::float32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 1;
        wp::float32* var_37;
        wp::int32 var_38;
        wp::float32 var_39;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::int32 var_42;
        wp::float32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        bool var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        const wp::int32 var_54 = 0;
        wp::int32 var_55;
        bool var_56;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::int32* var_60;
        const wp::int32 var_61 = 1;
        bool var_62;
        wp::int32 var_63;
        wp::int32* var_64;
        wp::float32* var_65;
        wp::int32 var_66;
        wp::float32 var_67;
        wp::float32* var_68;
        wp::float32 var_69;
        const wp::int32 var_70 = 2;
        wp::float32* var_71;
        wp::int32 var_72;
        wp::float32 var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::int32 var_76;
        wp::float32 var_77;
        wp::int32 var_78;
        const wp::int32 var_79 = 0;
        bool var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        bool var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        const wp::int32 var_86 = 0;
        bool var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        bool var_90;
        const wp::int32 var_91 = 1;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::int32* var_94;
        const wp::int32 var_95 = 1;
        bool var_96;
        wp::int32 var_97;
        wp::int32* var_98;
        wp::float32* var_99;
        wp::int32 var_100;
        wp::float32 var_101;
        wp::float32* var_102;
        wp::float32 var_103;
        const wp::int32 var_104 = 3;
        wp::float32* var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        wp::int32 var_108;
        wp::float32* var_109;
        wp::int32 var_110;
        wp::float32 var_111;
        wp::int32 var_112;
        const wp::int32 var_113 = 0;
        bool var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        bool var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        const wp::int32 var_120 = 0;
        bool var_121;
        const wp::int32 var_122 = 0;
        wp::int32 var_123;
        bool var_124;
        const wp::int32 var_125 = 1;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::int32* var_128;
        const wp::int32 var_129 = 1;
        bool var_130;
        wp::int32 var_131;
        wp::int32* var_132;
        wp::float32* var_133;
        wp::int32 var_134;
        wp::float32 var_135;
        wp::float32* var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 4;
        wp::float32* var_139;
        wp::int32 var_140;
        wp::float32 var_141;
        wp::int32 var_142;
        wp::float32* var_143;
        wp::int32 var_144;
        wp::float32 var_145;
        wp::int32 var_146;
        const wp::int32 var_147 = 0;
        bool var_148;
        wp::int32 var_149;
        wp::int32 var_150;
        bool var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        const wp::int32 var_154 = 0;
        bool var_155;
        const wp::int32 var_156 = 0;
        wp::int32 var_157;
        bool var_158;
        const wp::int32 var_159 = 1;
        wp::int32 var_160;
        wp::int32 var_161;
        wp::int32* var_162;
        const wp::int32 var_163 = 1;
        bool var_164;
        wp::int32 var_165;
        wp::int32* var_166;
        wp::float32* var_167;
        wp::int32 var_168;
        wp::float32 var_169;
        wp::float32* var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = 5;
        wp::float32* var_173;
        wp::int32 var_174;
        wp::float32 var_175;
        wp::int32 var_176;
        wp::float32* var_177;
        wp::int32 var_178;
        wp::float32 var_179;
        wp::int32 var_180;
        const wp::int32 var_181 = 0;
        bool var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        bool var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        const wp::int32 var_188 = 0;
        bool var_189;
        const wp::int32 var_190 = 0;
        wp::int32 var_191;
        bool var_192;
        const wp::int32 var_193 = 1;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::int32* var_196;
        const wp::int32 var_197 = 1;
        bool var_198;
        wp::int32 var_199;
        wp::int32* var_200;
        wp::float32* var_201;
        wp::int32 var_202;
        wp::float32 var_203;
        wp::float32* var_204;
        wp::float32 var_205;
        const wp::int32 var_206 = 6;
        wp::float32* var_207;
        wp::int32 var_208;
        wp::float32 var_209;
        wp::int32 var_210;
        wp::float32* var_211;
        wp::int32 var_212;
        wp::float32 var_213;
        wp::int32 var_214;
        const wp::int32 var_215 = 0;
        bool var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        bool var_219;
        wp::int32 var_220;
        wp::int32 var_221;
        const wp::int32 var_222 = 0;
        bool var_223;
        const wp::int32 var_224 = 0;
        wp::int32 var_225;
        bool var_226;
        const wp::int32 var_227 = 1;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::int32* var_230;
        const wp::int32 var_231 = 1;
        bool var_232;
        wp::int32 var_233;
        wp::int32* var_234;
        wp::float32* var_235;
        wp::int32 var_236;
        wp::float32 var_237;
        wp::float32* var_238;
        wp::float32 var_239;
        const wp::int32 var_240 = 7;
        wp::float32* var_241;
        wp::int32 var_242;
        wp::float32 var_243;
        wp::int32 var_244;
        wp::float32* var_245;
        wp::int32 var_246;
        wp::float32 var_247;
        wp::int32 var_248;
        const wp::int32 var_249 = 0;
        bool var_250;
        wp::int32 var_251;
        wp::int32 var_252;
        bool var_253;
        wp::int32 var_254;
        wp::int32 var_255;
        const wp::int32 var_256 = 0;
        bool var_257;
        const wp::int32 var_258 = 0;
        wp::int32 var_259;
        bool var_260;
        const wp::int32 var_261 = 1;
        wp::int32 var_262;
        wp::int32 var_263;
        wp::int32* var_264;
        const wp::int32 var_265 = 1;
        bool var_266;
        wp::int32 var_267;
        wp::int32* var_268;
        wp::float32* var_269;
        wp::int32 var_270;
        wp::float32 var_271;
        wp::float32* var_272;
        wp::float32 var_273;
        const wp::int32 var_274 = 8;
        wp::float32* var_275;
        wp::int32 var_276;
        wp::float32 var_277;
        wp::int32 var_278;
        wp::float32* var_279;
        wp::int32 var_280;
        wp::float32 var_281;
        wp::int32 var_282;
        const wp::int32 var_283 = 0;
        bool var_284;
        wp::int32 var_285;
        wp::int32 var_286;
        bool var_287;
        wp::int32 var_288;
        wp::int32 var_289;
        const wp::int32 var_290 = 0;
        bool var_291;
        const wp::int32 var_292 = 0;
        wp::int32 var_293;
        bool var_294;
        const wp::int32 var_295 = 1;
        wp::int32 var_296;
        wp::int32 var_297;
        wp::int32* var_298;
        const wp::int32 var_299 = 1;
        bool var_300;
        wp::int32 var_301;
        wp::int32* var_302;
        wp::float32* var_303;
        wp::int32 var_304;
        wp::float32 var_305;
        wp::float32* var_306;
        wp::float32 var_307;
        builtin_tid2d(var_0, var_1);
        var_3 = wp::address(var_cx, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::int(var_5);
        var_6 = wp::sub(var_0, var_4);
        var_7 = wp::address(var_cy, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::int(var_9);
        var_10 = wp::sub(var_1, var_8);
        var_12 = (var_6 < var_11);
        if (var_12) {
            var_13 = wp::add(var_6, var_nx);
        }
        var_14 = wp::where(var_12, var_13, var_6);
        var_15 = (var_14 >= var_nx);
        if (var_15) {
            var_16 = wp::sub(var_14, var_nx);
        }
        var_17 = wp::where(var_15, var_16, var_14);
        var_19 = (var_10 < var_18);
        if (var_19) {
        }
        var_21 = wp::where(var_19, var_20, var_10);
        var_22 = (var_21 >= var_ny);
        if (var_22) {
            var_24 = wp::sub(var_ny, var_23);
        }
        var_25 = wp::where(var_22, var_24, var_21);
        var_26 = wp::address(var_solid, var_17, var_25);
        var_29 = wp::load(var_26);
        var_28 = (var_29 == var_27);
        if (var_28) {
            var_30 = wp::address(var_opp, var_2);
            var_32 = wp::load(var_30);
            var_31 = wp::address(var_fpost, var_32, var_0, var_1);
            var_33 = wp::load(var_31);
            wp::array_store(var_f1, var_2, var_0, var_1, var_33);
        }
        if (!var_28) {
            var_34 = wp::address(var_fpost, var_2, var_17, var_25);
            var_35 = wp::load(var_34);
            wp::array_store(var_f1, var_2, var_0, var_1, var_35);
        }
        var_37 = wp::address(var_cx, var_36);
        var_39 = wp::load(var_37);
        var_38 = wp::int(var_39);
        var_40 = wp::sub(var_0, var_38);
        var_41 = wp::address(var_cy, var_36);
        var_43 = wp::load(var_41);
        var_42 = wp::int(var_43);
        var_44 = wp::sub(var_1, var_42);
        var_46 = (var_40 < var_45);
        if (var_46) {
            var_47 = wp::add(var_40, var_nx);
        }
        var_48 = wp::where(var_46, var_47, var_40);
        var_49 = (var_48 >= var_nx);
        if (var_49) {
            var_50 = wp::sub(var_48, var_nx);
        }
        var_51 = wp::where(var_49, var_50, var_48);
        var_53 = (var_44 < var_52);
        if (var_53) {
        }
        var_55 = wp::where(var_53, var_54, var_44);
        var_56 = (var_55 >= var_ny);
        if (var_56) {
            var_58 = wp::sub(var_ny, var_57);
        }
        var_59 = wp::where(var_56, var_58, var_55);
        var_60 = wp::address(var_solid, var_51, var_59);
        var_63 = wp::load(var_60);
        var_62 = (var_63 == var_61);
        if (var_62) {
            var_64 = wp::address(var_opp, var_36);
            var_66 = wp::load(var_64);
            var_65 = wp::address(var_fpost, var_66, var_0, var_1);
            var_67 = wp::load(var_65);
            wp::array_store(var_f1, var_36, var_0, var_1, var_67);
        }
        if (!var_62) {
            var_68 = wp::address(var_fpost, var_36, var_51, var_59);
            var_69 = wp::load(var_68);
            wp::array_store(var_f1, var_36, var_0, var_1, var_69);
        }
        var_71 = wp::address(var_cx, var_70);
        var_73 = wp::load(var_71);
        var_72 = wp::int(var_73);
        var_74 = wp::sub(var_0, var_72);
        var_75 = wp::address(var_cy, var_70);
        var_77 = wp::load(var_75);
        var_76 = wp::int(var_77);
        var_78 = wp::sub(var_1, var_76);
        var_80 = (var_74 < var_79);
        if (var_80) {
            var_81 = wp::add(var_74, var_nx);
        }
        var_82 = wp::where(var_80, var_81, var_74);
        var_83 = (var_82 >= var_nx);
        if (var_83) {
            var_84 = wp::sub(var_82, var_nx);
        }
        var_85 = wp::where(var_83, var_84, var_82);
        var_87 = (var_78 < var_86);
        if (var_87) {
        }
        var_89 = wp::where(var_87, var_88, var_78);
        var_90 = (var_89 >= var_ny);
        if (var_90) {
            var_92 = wp::sub(var_ny, var_91);
        }
        var_93 = wp::where(var_90, var_92, var_89);
        var_94 = wp::address(var_solid, var_85, var_93);
        var_97 = wp::load(var_94);
        var_96 = (var_97 == var_95);
        if (var_96) {
            var_98 = wp::address(var_opp, var_70);
            var_100 = wp::load(var_98);
            var_99 = wp::address(var_fpost, var_100, var_0, var_1);
            var_101 = wp::load(var_99);
            wp::array_store(var_f1, var_70, var_0, var_1, var_101);
        }
        if (!var_96) {
            var_102 = wp::address(var_fpost, var_70, var_85, var_93);
            var_103 = wp::load(var_102);
            wp::array_store(var_f1, var_70, var_0, var_1, var_103);
        }
        var_105 = wp::address(var_cx, var_104);
        var_107 = wp::load(var_105);
        var_106 = wp::int(var_107);
        var_108 = wp::sub(var_0, var_106);
        var_109 = wp::address(var_cy, var_104);
        var_111 = wp::load(var_109);
        var_110 = wp::int(var_111);
        var_112 = wp::sub(var_1, var_110);
        var_114 = (var_108 < var_113);
        if (var_114) {
            var_115 = wp::add(var_108, var_nx);
        }
        var_116 = wp::where(var_114, var_115, var_108);
        var_117 = (var_116 >= var_nx);
        if (var_117) {
            var_118 = wp::sub(var_116, var_nx);
        }
        var_119 = wp::where(var_117, var_118, var_116);
        var_121 = (var_112 < var_120);
        if (var_121) {
        }
        var_123 = wp::where(var_121, var_122, var_112);
        var_124 = (var_123 >= var_ny);
        if (var_124) {
            var_126 = wp::sub(var_ny, var_125);
        }
        var_127 = wp::where(var_124, var_126, var_123);
        var_128 = wp::address(var_solid, var_119, var_127);
        var_131 = wp::load(var_128);
        var_130 = (var_131 == var_129);
        if (var_130) {
            var_132 = wp::address(var_opp, var_104);
            var_134 = wp::load(var_132);
            var_133 = wp::address(var_fpost, var_134, var_0, var_1);
            var_135 = wp::load(var_133);
            wp::array_store(var_f1, var_104, var_0, var_1, var_135);
        }
        if (!var_130) {
            var_136 = wp::address(var_fpost, var_104, var_119, var_127);
            var_137 = wp::load(var_136);
            wp::array_store(var_f1, var_104, var_0, var_1, var_137);
        }
        var_139 = wp::address(var_cx, var_138);
        var_141 = wp::load(var_139);
        var_140 = wp::int(var_141);
        var_142 = wp::sub(var_0, var_140);
        var_143 = wp::address(var_cy, var_138);
        var_145 = wp::load(var_143);
        var_144 = wp::int(var_145);
        var_146 = wp::sub(var_1, var_144);
        var_148 = (var_142 < var_147);
        if (var_148) {
            var_149 = wp::add(var_142, var_nx);
        }
        var_150 = wp::where(var_148, var_149, var_142);
        var_151 = (var_150 >= var_nx);
        if (var_151) {
            var_152 = wp::sub(var_150, var_nx);
        }
        var_153 = wp::where(var_151, var_152, var_150);
        var_155 = (var_146 < var_154);
        if (var_155) {
        }
        var_157 = wp::where(var_155, var_156, var_146);
        var_158 = (var_157 >= var_ny);
        if (var_158) {
            var_160 = wp::sub(var_ny, var_159);
        }
        var_161 = wp::where(var_158, var_160, var_157);
        var_162 = wp::address(var_solid, var_153, var_161);
        var_165 = wp::load(var_162);
        var_164 = (var_165 == var_163);
        if (var_164) {
            var_166 = wp::address(var_opp, var_138);
            var_168 = wp::load(var_166);
            var_167 = wp::address(var_fpost, var_168, var_0, var_1);
            var_169 = wp::load(var_167);
            wp::array_store(var_f1, var_138, var_0, var_1, var_169);
        }
        if (!var_164) {
            var_170 = wp::address(var_fpost, var_138, var_153, var_161);
            var_171 = wp::load(var_170);
            wp::array_store(var_f1, var_138, var_0, var_1, var_171);
        }
        var_173 = wp::address(var_cx, var_172);
        var_175 = wp::load(var_173);
        var_174 = wp::int(var_175);
        var_176 = wp::sub(var_0, var_174);
        var_177 = wp::address(var_cy, var_172);
        var_179 = wp::load(var_177);
        var_178 = wp::int(var_179);
        var_180 = wp::sub(var_1, var_178);
        var_182 = (var_176 < var_181);
        if (var_182) {
            var_183 = wp::add(var_176, var_nx);
        }
        var_184 = wp::where(var_182, var_183, var_176);
        var_185 = (var_184 >= var_nx);
        if (var_185) {
            var_186 = wp::sub(var_184, var_nx);
        }
        var_187 = wp::where(var_185, var_186, var_184);
        var_189 = (var_180 < var_188);
        if (var_189) {
        }
        var_191 = wp::where(var_189, var_190, var_180);
        var_192 = (var_191 >= var_ny);
        if (var_192) {
            var_194 = wp::sub(var_ny, var_193);
        }
        var_195 = wp::where(var_192, var_194, var_191);
        var_196 = wp::address(var_solid, var_187, var_195);
        var_199 = wp::load(var_196);
        var_198 = (var_199 == var_197);
        if (var_198) {
            var_200 = wp::address(var_opp, var_172);
            var_202 = wp::load(var_200);
            var_201 = wp::address(var_fpost, var_202, var_0, var_1);
            var_203 = wp::load(var_201);
            wp::array_store(var_f1, var_172, var_0, var_1, var_203);
        }
        if (!var_198) {
            var_204 = wp::address(var_fpost, var_172, var_187, var_195);
            var_205 = wp::load(var_204);
            wp::array_store(var_f1, var_172, var_0, var_1, var_205);
        }
        var_207 = wp::address(var_cx, var_206);
        var_209 = wp::load(var_207);
        var_208 = wp::int(var_209);
        var_210 = wp::sub(var_0, var_208);
        var_211 = wp::address(var_cy, var_206);
        var_213 = wp::load(var_211);
        var_212 = wp::int(var_213);
        var_214 = wp::sub(var_1, var_212);
        var_216 = (var_210 < var_215);
        if (var_216) {
            var_217 = wp::add(var_210, var_nx);
        }
        var_218 = wp::where(var_216, var_217, var_210);
        var_219 = (var_218 >= var_nx);
        if (var_219) {
            var_220 = wp::sub(var_218, var_nx);
        }
        var_221 = wp::where(var_219, var_220, var_218);
        var_223 = (var_214 < var_222);
        if (var_223) {
        }
        var_225 = wp::where(var_223, var_224, var_214);
        var_226 = (var_225 >= var_ny);
        if (var_226) {
            var_228 = wp::sub(var_ny, var_227);
        }
        var_229 = wp::where(var_226, var_228, var_225);
        var_230 = wp::address(var_solid, var_221, var_229);
        var_233 = wp::load(var_230);
        var_232 = (var_233 == var_231);
        if (var_232) {
            var_234 = wp::address(var_opp, var_206);
            var_236 = wp::load(var_234);
            var_235 = wp::address(var_fpost, var_236, var_0, var_1);
            var_237 = wp::load(var_235);
            wp::array_store(var_f1, var_206, var_0, var_1, var_237);
        }
        if (!var_232) {
            var_238 = wp::address(var_fpost, var_206, var_221, var_229);
            var_239 = wp::load(var_238);
            wp::array_store(var_f1, var_206, var_0, var_1, var_239);
        }
        var_241 = wp::address(var_cx, var_240);
        var_243 = wp::load(var_241);
        var_242 = wp::int(var_243);
        var_244 = wp::sub(var_0, var_242);
        var_245 = wp::address(var_cy, var_240);
        var_247 = wp::load(var_245);
        var_246 = wp::int(var_247);
        var_248 = wp::sub(var_1, var_246);
        var_250 = (var_244 < var_249);
        if (var_250) {
            var_251 = wp::add(var_244, var_nx);
        }
        var_252 = wp::where(var_250, var_251, var_244);
        var_253 = (var_252 >= var_nx);
        if (var_253) {
            var_254 = wp::sub(var_252, var_nx);
        }
        var_255 = wp::where(var_253, var_254, var_252);
        var_257 = (var_248 < var_256);
        if (var_257) {
        }
        var_259 = wp::where(var_257, var_258, var_248);
        var_260 = (var_259 >= var_ny);
        if (var_260) {
            var_262 = wp::sub(var_ny, var_261);
        }
        var_263 = wp::where(var_260, var_262, var_259);
        var_264 = wp::address(var_solid, var_255, var_263);
        var_267 = wp::load(var_264);
        var_266 = (var_267 == var_265);
        if (var_266) {
            var_268 = wp::address(var_opp, var_240);
            var_270 = wp::load(var_268);
            var_269 = wp::address(var_fpost, var_270, var_0, var_1);
            var_271 = wp::load(var_269);
            wp::array_store(var_f1, var_240, var_0, var_1, var_271);
        }
        if (!var_266) {
            var_272 = wp::address(var_fpost, var_240, var_255, var_263);
            var_273 = wp::load(var_272);
            wp::array_store(var_f1, var_240, var_0, var_1, var_273);
        }
        var_275 = wp::address(var_cx, var_274);
        var_277 = wp::load(var_275);
        var_276 = wp::int(var_277);
        var_278 = wp::sub(var_0, var_276);
        var_279 = wp::address(var_cy, var_274);
        var_281 = wp::load(var_279);
        var_280 = wp::int(var_281);
        var_282 = wp::sub(var_1, var_280);
        var_284 = (var_278 < var_283);
        if (var_284) {
            var_285 = wp::add(var_278, var_nx);
        }
        var_286 = wp::where(var_284, var_285, var_278);
        var_287 = (var_286 >= var_nx);
        if (var_287) {
            var_288 = wp::sub(var_286, var_nx);
        }
        var_289 = wp::where(var_287, var_288, var_286);
        var_291 = (var_282 < var_290);
        if (var_291) {
        }
        var_293 = wp::where(var_291, var_292, var_282);
        var_294 = (var_293 >= var_ny);
        if (var_294) {
            var_296 = wp::sub(var_ny, var_295);
        }
        var_297 = wp::where(var_294, var_296, var_293);
        var_298 = wp::address(var_solid, var_289, var_297);
        var_301 = wp::load(var_298);
        var_300 = (var_301 == var_299);
        if (var_300) {
            var_302 = wp::address(var_opp, var_274);
            var_304 = wp::load(var_302);
            var_303 = wp::address(var_fpost, var_304, var_0, var_1);
            var_305 = wp::load(var_303);
            wp::array_store(var_f1, var_274, var_0, var_1, var_305);
        }
        if (!var_300) {
            var_306 = wp::address(var_fpost, var_274, var_289, var_297);
            var_307 = wp::load(var_306);
            wp::array_store(var_f1, var_274, var_0, var_1, var_307);
        }
    }
}



