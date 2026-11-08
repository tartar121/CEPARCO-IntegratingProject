; xmm.asm
; ASUM using 128-bit XMM registers (AVX)
; computes asum = sum of |A[i]|
; input: rcx = pointer to A, edx = number of elements n
; output: xmm0 = sum of absolute values
section .data
    negate dq -1.0

section .text
bits 64
default rel
global xmmkernel

xmmkernel:
    vxorpd xmm0, xmm0, xmm0
    xor r9, r9

    cmp edx, 0
    jle finish
    mov edx, edx

    vmovddup xmm3, [negate]

process_pairs:
    mov r10, rdx
    sub r10, r9
    cmp r10, 2
    jb reduce

    vmovupd xmm1, [rcx + r9*8]
    vmulpd xmm2, xmm1, xmm3
    vmaxpd xmm1, xmm1, xmm2
    vaddpd xmm0, xmm0, xmm1

    add r9, 2
    jmp process_pairs

reduce:
    vhaddpd xmm0, xmm0, xmm0

remainder_loop:
    cmp r9, rdx
    jae finish

    movsd xmm1, [rcx + r9*8]
    vxorpd xmm2, xmm2, xmm2
    ucomisd xmm1, xmm2
    jae add_value
    mulsd xmm1, [negate]

add_value:
    addsd xmm0, xmm1
    inc r9
    jmp remainder_loop

finish:
    ret