'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/Thegolfingocto/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



from ADFNNTrainer import ADFNNTrainer
import copy

def AutoAddTrainers(T: ADFNNTrainer, strKey: str, vecValues: list[str]) -> list[ADFNNTrainer]:
    if T.GetValue(strKey) in vecValues: vecValues.remove(T.GetValue(strKey))
    vecT = [T]
    for strValue in vecValues:
        newCfg = copy.deepcopy(T.dCfg)
        newCfg[strKey] = strValue
        vecT.append(ADFNNTrainer("../ADFNNTrainer", newCfg))

    return vecT