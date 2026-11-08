; ymm.asm
; ASUM using 256-bit YMM registers (AVX)
; computes asum = sum of |A[i]|
; input: rcx = pointer to A, edx = number of elements n
; output: xmm0 = sum of absolute values
section .data
    negate dq -1.0

section .text
bits 64
default rel
global ymmkernel

ymmkernel:
    vxorps ymm0, ymm0, ymm0
    xor r9, r9

    vbroadcastsd ymm3, [negate] ; broadcast -1.0 to all four double

    cmp edx, 0
    jle finish
    mov edx, edx

process_groups:
    ; process four doubles if at least four remain
    mov r10, rdx
    sub r10, r9
    cmp r10, 4
    jb remainder

    vmovupd ymm1, [rcx + r9*8]
    vmulpd ymm2, ymm1, ymm3
    vmaxpd ymm1, ymm1, ymm2
    vaddpd ymm0, ymm0, ymm1

    add r9, 4
    jmp process_groups

remainder:
    ; reduce the four accumulated
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    vhaddpd xmm0, xmm0, xmm0

remainder_loop:
    ; process remaining elements one at a time
    cmp r9, rdx
    jae finish

    movsd xmm1, [rcx + r9*8]
    vxorps xmm2, xmm2, xmm2
    ucomisd xmm1, xmm2
    jnc add_value

    mulsd xmm1, [negate]

add_value:
    addsd xmm0, xmm1
    inc r9
    jmp remainder_loop

finish:
    vzeroupper ; clear upper YMM state; preserve XMM0 return value
    ret