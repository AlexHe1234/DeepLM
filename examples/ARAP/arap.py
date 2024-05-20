import argparse

from ARAPProblem.io import *
from ARAPProblem.loss import *

from TorchLM.solver import SolveFunc, FunctionBlock
from time import time

from knn_cuda import KNN
from fpsample import bucket_fps_kdline_sampling


def knn(ref: torch.Tensor, qry: torch.Tensor, k: int):
    # ref B, T, 3
    # qry B, N, 3
    assert torch.cuda.is_available(), 'knn requires cuda'
    knn = KNN(k=k, transpose_mode=True)
    dist, index = knn(ref, qry)  # both of B, N, 4
    return dist, index


def reg(src, tar, knn, fps):  # np array
	# V, F = LoadOBJ(args.inputFile)
	# breakpoint()
 
	V = src
	vc = np.mean(V, axis=0)
 
	for i in range(3):
		V[:,i] -= vc[i]

	# SaveOBJ('obj/origin.obj', V, F)

	V = torch.from_numpy(V)
	# F = torch.from_numpy(F)

	VFrames = torch.zeros((V.shape[0], 3)).double()

	device = 'cuda'
	V, VFrames = V.to(device), VFrames.to(device)

	# for each node find nearest neighbors to stay rigid
	dist, index = knn(V[None], V[None], knn)  # index B, N, K
	index = index[0]  # N, K
	index0 = torch.arange(index.shape[0])[None].repeat(index.shape[1], 1).T  # N, K
	
	e1Idx = index0
	e2Idx = index

	srcIdx = torch.cat((e1Idx, e2Idx))
	tarIdx = torch.cat((e2Idx, e1Idx))

	originOffset = V[tarIdx] - V[srcIdx]

	vIndices = torch.cat((srcIdx.view(-1,1), tarIdx.view(-1,1)), dim=1)

	rigidityFunc = FunctionBlock(variables = [V, VFrames],
		constants = [originOffset],
		indices = [vIndices, srcIdx],
		fn = RigidityError)

	# build controls
	# controlIdx = torch.from_numpy(np.array([
	# 	2,5000,10000,15000,20000,25000,30000])).long()
	controlIdx = torch.from_numpy(bucket_fps_kdline_sampling(src, fps, 5)).long()
	controlIdx = controlIdx.to(device)
	targetPt = V[controlIdx].clone()

	py = targetPt[0,2].clone()
	theta = 30.0 / 180.0 * np.pi
	targetPt[0,0] = np.sin(theta) * py
	targetPt[0,2] = np.cos(theta) * py

	distanceFunc = FunctionBlock(variables = [V],
		constants = [targetPt],
		indices = [controlIdx],
		fn = DistanceError)

	t1 = time()

	SolveFunc(funcs = [rigidityFunc, distanceFunc],
		numIterations = 25,
		numSuccessIterations = 25)

	t2 = time()

	print("Time used %f secs."%(t2 - t1))

	return V.data
	V = V.data.cpu().numpy()
	F = F.data.cpu().numpy()
	SaveOBJ(args.outputFile, V, F)
