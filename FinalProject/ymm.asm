section .data
msg db "YMM Test",13,10,0

section .text
bits 64
default rel

global ymmkernel
extern printf

;ymmkernel:
	;sub rsp, 8*5
	;lea rcx, [msg]
	;lea rcx, [msg]
	;call printf
	;add rsp, 8*5
	;ret

; ======================================================================
; YMM (AVX, 256-bit, VEX-encoded) assignment kernel for k-means.
; ======================================================================

section .data
align 32
one8 dd 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0   ; loop-invariant index increment

section .text
global ymm_assign_kernel
extern assign_tail_c            ; same C callee as the XMM kernel (kmeans.c)

ymm_assign_kernel:
	push rbx
	push rsi
	push rdi

	mov r10, [rsp+64]           ; labels
	mov r11, [rsp+72]           ; min_d2
	mov eax, [rsp+80]           ; N
	mov esi, [rsp+88]           ; K

	sub rsp, 96                 ; 64: shadow space + 4 stack args for the C call, 32: xmm6/xmm7 save
	vmovaps [rsp+64], xmm6      ; xmm6/xmm7 are callee-saved (only their low 128 bits matter in Win64)
	vmovaps [rsp+80], xmm7

	test esi, esi
	jle .done                   ; K < 1: nothing to do

	mov edi, eax
	and edi, 7                  ; edi = leftover count (N % 8)
	shr eax, 3                  ; eax = number of 8-point blocks
	jz  .tail                   ; N < 8: everything goes to the C callee

	vmovups ymm7, [one8]        ; ymm7 = {1.0 x8}

.block:
	vmovups ymm0, [rcx]         ; px = X[i..i+7]
	vmovups ymm1, [rdx]         ; py = Y[i..i+7]

	; centroid 0 initialises the running minimum
	vbroadcastss ymm4, [r8]     ; cx[0] in all lanes
	vsubps ymm4, ymm0, ymm4     ; dx
	vmulps ymm4, ymm4, ymm4     ; dx*dx
	vbroadcastss ymm5, [r9]     ; cy[0]
	vsubps ymm5, ymm1, ymm5     ; dy
	vmulps ymm5, ymm5, ymm5     ; dy*dy
	vaddps ymm2, ymm4, ymm5     ; ymm2 = best d2
	vxorps ymm3, ymm3, ymm3     ; ymm3 = best index (0.0 in every lane)

	mov ebx, 1                  ; j = 1
	cmp ebx, esi
	jge .store                  ; K == 1: centroid 0 is the only candidate
	vmovaps ymm6, ymm7          ; ymm6 = {j,j,...} = {1.0 x8}
.cloop:
	vbroadcastss ymm4, [r8+rbx*4]
	vsubps ymm4, ymm0, ymm4
	vmulps ymm4, ymm4, ymm4
	vbroadcastss ymm5, [r9+rbx*4]
	vsubps ymm5, ymm1, ymm5
	vmulps ymm5, ymm5, ymm5
	vaddps ymm4, ymm4, ymm5     ; ymm4 = d2 for centroid j

	vcmpltps ymm5, ymm4, ymm2   ; mask = d2 < best (strict, so ties keep the lower index)
	vminps ymm2, ymm4, ymm2     ; best = (d2 < best) ? d2 : best
	vblendvps ymm3, ymm3, ymm6, ymm5   ; idx = mask ? j : idx
	vaddps ymm6, ymm6, ymm7     ; j vector += 1.0

	inc ebx
	cmp ebx, esi
	jl .cloop                   ; single bottom-tested branch

.store:
	vcvttps2dq ymm3, ymm3       ; float index -> int32 label
	vmovups [r10], ymm3         ; labels[i..i+7]
	vmovups [r11], ymm2         ; min_d2[i..i+7]

	add rcx, 32
	add rdx, 32
	add r10, 32
	add r11, 32
	dec eax
	jnz .block

.tail:
	; leftover N % 8 points: asm calls the C callee.
	; rcx/rdx are already at the first leftover point of X and Y,
	; r10/r11 at the first leftover slot of labels and min_d2, r8/r9 = cx/cy.
	test edi, edi
	jz  .done

	mov [rsp+32], r10           ; arg5: labels (tail)
	mov [rsp+40], r11           ; arg6: min_d2 (tail)
	mov [rsp+48], rdi           ; arg7: count (upper half is zero)
	mov [rsp+56], rsi           ; arg8: K     (upper half is zero)
	vzeroupper
	call assign_tail_c

.done:
	vmovaps xmm6, [rsp+64]      ; restore callee-saved xmm6/xmm7
	vmovaps xmm7, [rsp+80]
	add rsp, 96
	pop rdi
	pop rsi
	pop rbx
	vzeroupper
	ret