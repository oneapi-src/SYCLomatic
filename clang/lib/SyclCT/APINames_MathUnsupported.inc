// Half Arithmetic Functions
ENTRY("__hadd_sat")
ENTRY("__hfma_sat")
ENTRY("__hmul_sat")
ENTRY("__hsub_sat")

// Half2 Arithmetic Functions
ENTRY("__hadd2_sat")
ENTRY("__hfma2_sat")
ENTRY("__hmul2_sat")
ENTRY("__hsub2_sat")

// Half Comparison Functions
ENTRY("__hequ")
ENTRY("__hgeu")
ENTRY("__hgtu")
ENTRY("__hleu")
ENTRY("__hltu")
ENTRY("__hneu")

// Half2 Comparison Functions
ENTRY("__hbeq2")
ENTRY("__hbequ2")
ENTRY("__hbge2")
ENTRY("__hbgeu2")
ENTRY("__hbgt2")
ENTRY("__hbgtu2")
ENTRY("__hble2")
ENTRY("__hbleu2")
ENTRY("__hblt2")
ENTRY("__hbltu2")
ENTRY("__hbne2")
ENTRY("__hbneu2")
ENTRY("__heq2")
ENTRY("__hequ2")
ENTRY("__hge2")
ENTRY("__hgeu2")
ENTRY("__hgt2")
ENTRY("__hgtu2")
ENTRY("__hisnan2")
ENTRY("__hle2")
ENTRY("__hleu2")
ENTRY("__hlt2")
ENTRY("__hltu2")
ENTRY("__hne2")
ENTRY("__hneu2")

// Half Math Functions
ENTRY("hrcp")

// Half2 Math Functions
ENTRY("h2rcp")

// Single Precision Mathematical Functions
ENTRY("cyl_bessel_i0f")
ENTRY("cyl_bessel_i1f")
ENTRY("erfcinvf")
ENTRY("erfcxf")
ENTRY("erfinvf")
ENTRY("j0f")
ENTRY("j1f")
ENTRY("jnf")
ENTRY("norm3df")
ENTRY("norm4df")
ENTRY("normcdff")
ENTRY("normcdfinvf")
ENTRY("normf")
ENTRY("rcbrtf")
ENTRY("rnorm3df")
ENTRY("rnorm4df")
ENTRY("rnormf")
ENTRY("scalblnf")
ENTRY("scalbnf")
ENTRY("y0f")
ENTRY("y1f")
ENTRY("ynf")

// Double Precision Mathematical Functions
ENTRY("cyl_bessel_i0")
ENTRY("cyl_bessel_i1")
ENTRY("erfcinv")
ENTRY("erfcx")
ENTRY("erfinv")
ENTRY("j0")
ENTRY("j1")
ENTRY("jn")
ENTRY("norm")
ENTRY("norm3d")
ENTRY("norm4d")
ENTRY("normcdf")
ENTRY("normcdfinv")
ENTRY("rcbrt")
ENTRY("rnorm3d")
ENTRY("rnorm4d")
ENTRY("rnorm")
ENTRY("scalbln")
ENTRY("scalbn")
ENTRY("y0")
ENTRY("y1")
ENTRY("yn")

// Single precision intrinisics
ENTRY("__saturatef")

// Double precision intrinisics
ENTRY("__drcp_rd")
ENTRY("__drcp_rn")
ENTRY("__drcp_ru")
ENTRY("__drcp_rz")

// Half Precision Conversion And Data Movement
ENTRY("__shfl_down_sync")
ENTRY("__shfl_sync")
ENTRY("__shfl_up_sync")
ENTRY("__shfl_xor_sync")

// Type Casting Intrinsics
ENTRY("__double2hiint") // TODO
ENTRY("__double2loint") // TODO
ENTRY("__hiloint2double") // TODO

// Integer Intrinsics
ENTRY("__brev")
ENTRY("__brevll")
ENTRY("__byte_perm")
ENTRY("__ffs")
ENTRY("__ffsll")
ENTRY("__funnelshift_l")
ENTRY("__funnelshift_lc")
ENTRY("__funnelshift_r")
ENTRY("__funnelshift_rc")
ENTRY("__mul64hi")
ENTRY("__rhadd")
ENTRY("__sad")
ENTRY("__uhadd")
ENTRY("__umul24")
ENTRY("__umul64hi")
ENTRY("__umulhi")
ENTRY("__urhadd")
ENTRY("__usad")
