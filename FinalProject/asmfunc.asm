section .data
    negate dq -1.0 ; value to make negative number to positive 
    clear dq 0.0

section .text
bits 64
default rel
global asmkernel

asmkernel:
		xor r9, r9
		movsd xmm0, [clear]
		asm_loop:
		cmp r9, rdx
		JE done
		movsd xmm1, [rcx+r9*8]
		movsd xmm2, [clear]
		ucomisd xmm1, xmm2    ; if less than 0  CF = 1
		JNC adding
		vmulsd xmm1, [negate]
		adding:
		addsd xmm0, xmm1
		inc r9
		JMP asm_loop
		
		done:
		ret
