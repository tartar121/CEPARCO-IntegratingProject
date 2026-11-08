section .data
msg db "YMM Test",13,10,0

section .text
bits 64
default rel

global ymmkernel
extern printf

ymmkernel:
	sub rsp, 8*5
	lea rcx, [msg]
	call printf
	add rsp, 8*5
	ret