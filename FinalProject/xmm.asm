section .data
msg db "XMM Test",13,10,0

section .text
bits 64
default rel

global xmmkernel
extern printf

xmmkernel:
	sub rsp, 8*5
	lea rcx, [msg]
	call printf
	add rsp, 8*5
	ret