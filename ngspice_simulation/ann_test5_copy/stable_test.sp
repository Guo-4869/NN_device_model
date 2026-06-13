* single_point.sp
* 单点测试



.control
pre_osdi stable_bsim_nn.osdi
.endc

.model NMOS1 bsim_nn(w=10e-6 l=1e-6)

M1 d g 0 0 NMOS1
Vds d 0 0.5
Vgs g 0 0.5

.op

.control
run
print "ID = " i(vds) "A"
print "log10(ID) = " log10(i(vds))
.endc

.end