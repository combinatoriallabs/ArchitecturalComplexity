'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



import torch
import time

device = "cpu" if not torch.cuda.is_available() else "cuda"


def FishProduct(A: torch.tensor, B: torch.tensor, C: torch.tensor) -> torch.tensor:
    if len(A.shape) == 3: A = A.unsqueeze(0)
    if len(B.shape) == 3: B = B.unsqueeze(0)
    if len(C.shape) == 3: C = C.unsqueeze(0)

    assert A.shape[0] == C.shape[0], "Fish Product does not support batched parameters!"
    #assert B.shape[-2] == C.shape[1] and B.shape[-1] == C.shape[2], "Invalid tensor shapes: {}, {}, {}".format(A.shape, B.shape, C.shape)

    sB = list(B.shape)
    sC = list(C.shape)

    B = B.reshape(sB[:-2] + [-1])
    C = C.reshape(sC[:-3] + [-1] + [sC[-1]]) #3
    #print("B:", B.shape)
    #print("C:", C.shape)
    
    T = B @ C

    T = T.unsqueeze(-3)
    #print("A: ", A.shape, "T: ", T.shape)

    R = A @ T
    #print("R: ", R.shape)

    return R

def HalfMatMul(A: torch.tensor, B: torch.tensor, strReduction: str = "none") -> torch.tensor:
    '''
    iReduction: {-1, 1, 2} = -1: None, 1: Sum, 2: Prod
    '''
    assert A.shape[-2] == A.shape[-1], "HalfMatMul() only supports square matrices! {}, {}".format(A.shape, B.shape)

    iSz = A.shape[-1]

    X = torch.cat((A, A.transpose(-2, -1)), -2)

    #print("X:", X.shape, "Y:", Y.shape)

    R = (X @ B).unfold(-2, iSz, iSz)
    #print("R:", R.shape)

    if strReduction.lower() == "sum": return torch.sum(R, -3)
    elif strReduction.lower() == "prod": return torch.prod(R, -3)

    return R

def HalfMatMulType2(A: torch.tensor, B: torch.tensor, strReduction: str = "none") -> torch.tensor:
    '''
    iReduction: {-1, 1, 2} = -1: None, 1: Sum, 2: Prod
    '''
    assert B.shape[-2] == B.shape[1], "HalfMatMulType2() only supports square matrices! {}, {}".format(A.shape, B.shape)

    iSz = A.shape[-1]

    Y = torch.cat((B, B.transpose(-2, -1)), -1)

    #print("X:", X.shape, "Y:", Y.shape)

    R = (A @ Y).unfold(-1, iSz, iSz)
    #print("R:", R.shape)

    if strReduction.lower() == "sum": return torch.sum(R, -2)
    elif strReduction.lower() == "prod": return torch.prod(R, -2)

    return R

def FullMatMul(A: torch.tensor, B: torch.tensor, strReduction: str = "none") -> torch.tensor:
    '''
    iReduction: {-1, 1, 2} = -1: None, 1: Sum, 2: Prod
    '''
    assert A.shape[-2] == A.shape[-1] and B.shape[-2] == B.shape[1], "FullMatMul() only supports square matrices! {}, {}".format(A.shape, B.shape)

    iSz = A.shape[-1]

    X = torch.cat((A, A.transpose(-2, -1)), -2)
    Y = torch.cat((B, B.transpose(-2, -1)), -1)

    #print("X:", X.shape, "Y:", Y.shape)

    R = (X @ Y).unfold(-2, iSz, iSz).unfold(-2, iSz, iSz)
    #print("R:", R.shape)

    if strReduction.lower() == "sum": return torch.sum(torch.sum(R, -3), -3)
    elif strReduction.lower() == "prod": return torch.prod(torch.prod(R, -3), -3)

    return R


def ConeProductTorch(A: torch.tensor, B: torch.tensor, C: torch.tensor) -> torch.tensor:
    #Require batched tensors
    if len(A.shape) == 3: A = A.unsqueeze(0)
    if len(B.shape) == 3: B = B.unsqueeze(0)
    if len(C.shape) == 3: C = C.unsqueeze(0)

    #check batch dims?
    #assert A.shape[0] == B.shape[0] and B.shape[0] == C.shape[0], "Invalid batch dimensions: {}, {}, {}".format(A.shape[0], B.shape[0], C.shape[0])
    #check for contraction mode compat.
    assert A.shape[-1] == B.shape[-1] and B.shape[-1] == C.shape[-1], "Invalid tensor shapes: {}, {}, {}".format(A.shape, B.shape, C.shape)
    #check for output mode compat.
    assert A.shape[-3] == B.shape[-3] and B.shape[-2] == C.shape[-2] and A.shape[-2] == C.shape[-3], "Invalid tensor shapes: {}, {}, {}".format(A.shape, B.shape, C.shape)

    # A = A.to(device)
    # B = B.to(device)
    # C = C.to(device)

    #I = A.shape[1]
    #J = A.shape[2]
    #K = B.shape[2]

    A = A.unsqueeze(-2)
    #A = A.expand(-1, -1, -1, K, -1)

    B = B.unsqueeze(-3)
    #B = B.expand(-1, -1, J, -1, -1)

    C = C.unsqueeze(-4)
    #C = C.expand(-1, I, -1, -1, -1)

    #print(A.shape, B.shape, C.shape)

    return torch.sum(A * B * C, dim = -1)#.to("cpu")


def DTAProduct(tStruct: torch.tensor, tContract: torch.tensor, vecOperands: list[torch.tensor], vecContractionTiming: list[torch.tensor] = None) -> torch.tensor:
    '''
    Assumes any batch modes are placed at the start.
    Explicitly pass vecContractionTiming (of length arity-1) to avoid dynamically computing it
    '''
    iA = tStruct.shape[0]
    iOC = tStruct.shape[1]

    assert iA == len(vecOperands), "Incorrent amount of operands provided"
    iMaxOffset = 0
    for i in range(iA):
        iOrder = len(vecOperands[i].shape)
        iOffset = int(max([iOrder - torch.sum(tStruct[i, :]), 0]))
        if iOffset > iMaxOffset: iMaxOffset = iOffset
        
        for j in range(iOC):
            if not tStruct[i, j]: vecOperands[i] = vecOperands[i].unsqueeze(iOffset + j)

    #print(tStruct)
    #for i in range(iA): print(vecOperands[i].shape)

    #vecC = list(iMaxOffset + torch.where(tContract == 1)[0])
    #vecTotalC = copy.deepcopy(vecC)

    R = vecOperands[0]
    #print("Initial shape:", R.shape)

    if vecContractionTiming is None:
        for i in range(1, iA):
            R = R * vecOperands[i]

            #print("Shape after {}th hadamard:".format(i), R.shape)

            #find the "effectively free" contractions and process them ASAP to save memory
            tProcessableC = iMaxOffset + torch.where(torch.where(torch.sum(tStruct[i+1:, :], dim = 0) == 0, 1, 0) * tContract == 1)[0]
            #print(tProcessableC)
            #print(tProcessableC)
            if tProcessableC.shape[0] > 0: R = torch.sum(R, dim = tuple(tProcessableC), keepdim = True)

            #print("Shape after {}th contraction check:".format(i), R.shape)
            #input()
    else:
        for i in range(1, iA):
            R = R * vecOperands[i]
            #print(tuple(iMaxOffset + vecContractionTiming[i - 1]))
            if vecContractionTiming[i - 1].shape[0] > 0: R = torch.sum(R, dim = tuple(iMaxOffset + vecContractionTiming[i - 1]), keepdim = True)
            #print("Shape after {}th contraction check:".format(i), R.shape)
    
    #print(R.shape)
    vecPerm =  [i for i in range(iMaxOffset)] + list(torch.argsort(tContract) + iMaxOffset)
    #print(vecPerm)
    iC = int(torch.sum(tContract))
    #print("iC:", iC)
    R = R.permute(vecPerm)
    vecS = list(R.shape)
    #print(R.shape, vecS)
    R = R.reshape(vecS[:-1*iC])
    # R = torch.sum(R, dim = -1)

    return R



if __name__ == "__main__":
    # X = torch.randn((128, 30, 30, 3, 3, 3)) #thinking of this as an image
    # W1 = torch.randn((16, 1, 3)) #2x2 grid of channel-wise weights
    # W2 = torch.randn((3, 3, 1)) #size 2 vector of 8x8 weights

    # X = torch.randn((128, 64, 3, 4, 4))
    # W1 = torch.randn((3, 4, 3))
    # W2 = torch.randn((4, 4, 4))

    #FishProduct(W1, X, W2)

    # A = torch.randn((128, 65, 65))
    # B = torch.randn((128, 65, 65))

    # R = HalfMatMulType2(A, B)
    # print(R.shape)


    #Test for Cone Product
    # A = torch.randn((128, 50, 45, 32)).to(device)
    # B = torch.randn((128, 50, 40, 32)).to(device)
    # C = torch.randn((128, 45, 40, 32)).to(device)

    # tS1 = time.time()
    # R = ConeProductTorch(A, B, C)
    # tE1 = time.time()
    # print(R.shape)

    # tConeStruct = torch.tensor([
    #     [1, 1, 1, 0, 1],
    #     [1, 1, 0, 1, 1],
    #     [1, 0, 1, 1, 1],

    #     [0, 0, 0, 0, 1]
    # ])

    #Fish
    A = torch.randn((31, 30, 3)).to(device)
    B = torch.randn((128, 3, 32, 32)).to(device)
    C = torch.randn((32, 32, 4)).to(device)

    tFishStruct = torch.tensor([
        [0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],

        [0, 0, 0, 1, 1, 1, 0]
    ])

    tS1 = time.time()
    R = FishProduct(A, B, C)
    tE1 = time.time()
    print(R.shape)

    tS2 = time.time()
    R2 = DTAProduct(tFishStruct[:-1, :], tFishStruct[-1, :], [A, B, C])
    tE2 = time.time()
    print(R2.shape)

    print("Times:", tE1 - tS1, tE2 - tS2)
    print("Error:", torch.sum((R2 - R)**2))