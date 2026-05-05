'''
Code written by Nicholas J. Cooper.
Released under the MIT license, see the GitHub page for the full legal boilerplate.
tldr: you freely can do whatever you like with this code so long as this message is retained and you cite the GitHub: https://github.com/combinatoriallabs/ArchitecturalComplexity
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
'''



from ADFNNTrainer import *
from ADFNNSearchParams import *


@dataclass
class SearchParams:

    iTOMsPerModel: int

    iGiveUpE: int

    fUnaryOpProbability: float = 0.5

def ProcessResult(dResult: dict, dcParams: SearchParams) -> dict:
    '''
    Expects "standard" format results from an ITrainer
    '''
    if len(dResult["Runs"]) == 0: return None

    dRet = {
        "Params": dResult["Parameters"],
        "Runs": dResult["Runs"]
    }
    dR = dResult["Runs"][0]
    dRet["Time"] = dR["TotalTime"] / dResult["NumEpochs"]

    iTrLossIdx = dResult["TrainMetricNames"].index("TrainLoss")
    iTrAccIdx = dResult["TrainMetricNames"].index("TrainAcc")

    iTsLossIdx = dResult["TestMetricNames"].index("TestLoss")
    iTsAccIdx = dResult["TestMetricNames"].index("TestAcc")

    fMaxTrAcc = max([v[iTrAccIdx] for v in dR["TrainMetrics"]])
    fMaxTsAcc = max([v[iTsAccIdx] for v in dR["TestMetrics"]])

    fMinTrLoss = min([v[iTrLossIdx] for v in dR["TrainMetrics"]])
    fMinTsLoss = min([v[iTsLossIdx] for v in dR["TestMetrics"]])

    dRet["TestAcc"] = fMaxTsAcc
    dRet["AvgTestAcc"] = dResult["AvgTestAcc"]
    dRet["StdTestAcc"] = dResult["StdTestAcc"]
    if len(dR["TestMetrics"]) > dcParams.iGiveUpE - 1:
        dRet["TestAccAtThreshold"] = dR["TestMetrics"][dcParams.iGiveUpE - 1][iTsAccIdx]
    else: dRet["TestAccAtThreshold"] = 1.0
    dRet["TestLoss"] = fMinTsLoss
    dRet["GenGapAcc"] = fMaxTrAcc - fMaxTsAcc
    dRet["GenGapLoss"] = fMinTsLoss - fMinTrLoss

    return dRet



def PlotADFNNSearch(dCfg: dict, strDir: str, dcParams: SearchParams) -> None:
    T = ADFNNTrainer(strDir, dCfg)

    dResults = {}
    def AddResult(strHashDir: str) -> None:
        with open(strHashDir + "Result.json", "r") as f: dResults[strHashDir.split("/")[-2]] = ProcessResult(json.load(f), dcParams)
        if not os.path.exists(strHashDir + "Model.pth"): print(f"NOTICE: No model found for {strHashDir}")
        return
    T.FindCachedStuff("Result.json", cFoundItemCB = AddResult)

    mapLabelsToIDs = {}
    def PickPrinter(mplEvent):
        strL = mplEvent.artist.get_label()
        strID = mapLabelsToIDs[strL]
        strModelID = T.CacheMap[strID]["ADFNNModelID"]
        T.cacheADFNN.PrintADFNN(strModelID)
        print("Test Acc: {:.2f}, Gen Gap: {:.2f}".format(100 * dResults[strID]["TestAcc"], 100 * dResults[strID]["GenGapAcc"]))
        print("Stacks: {}, Params: {}, Time: {}".format(T.CacheMap[strID]["ADFNNStacks"], dResults[strID]["Params"], dResults[strID]["Time"]))
        print("ADFNN ID: ", strModelID)
        print("---------------------------------")
        
        return

    vecColors = ["midnightblue", "dodgerblue", "skyblue", "maroon", "orange", "darkcyan", "green", "orchid"]
    vecMarkers = ["v", "s", "*"]

    fig, ax = plt.subplots()

    def ScatterTestBench():

        vecX = []
        vecY = []
        mapNumTOMsToSampleCount = {}
        mapNumTensToSampleCount = {}
        mapNumNLToSampleCount = {}
        mapAToSampleCount = {}
        mapCAToSampleCount = {}
        mapOCToSampleCount = {}

        for strID in dResults.keys():
            bSkip = False
            bSkip = bSkip or (T.CacheMap[strID]["Dataset"] != dCfg["Dataset"])

            bSkip = bSkip or (T.CacheMap[strID]["Model"] != "ADFNN")
            bSkip = bSkip or (T.CacheMap[strID]["ADFNNInputModule"] != dCfg["ADFNNInputModule"])

            #filter based on training recipe
            bSkip = bSkip or (T.CacheMap[strID]["NumEpochs"] != dCfg["NumEpochs"])
            bSkip = bSkip or (T.CacheMap[strID]["DataAugmentation"] != dCfg["DataAugmentation"])
            bSkip = bSkip or (T.CacheMap[strID]["LRScheduler"] != dCfg["LRScheduler"])
            bSkip = bSkip or (T.CacheMap[strID]["SubSample"] != dCfg["SubSample"])

            #print(strID, T.CacheMap[strID]["BatchSize"])

            #lin-sep. threshold
            bSkip = bSkip or (len(dResults[strID]["Runs"][0]["TestMetrics"]) <= dcParams.iGiveUpE)


            strADFNNID = T.CacheMap[strID]["ADFNNModelID"]
            dModelCfg = T.cacheADFNN.CacheMap[strADFNNID]

            vecOC = [len(vecS) for vecS in dModelCfg["TOMShapes"]]
            avgOC = sum(vecOC) / len(vecOC)
            sOC = sum(vecOC)

            vecIAD = [sum(vecS) for vecS in dModelCfg["TOMShapes"]]
            avgIAD = sum(vecIAD) / len(vecIAD)

            tTEMStruct = T.cacheADFNN.GetTEM(dModelCfg["TEMID"])
            if tTEMStruct is None:
                print("WARNING: missing TEM {} from cfg {}".format(dModelCfg["TEMID"], strID))
                input()
                continue
            tTEM = TEMatrix(tTEMStruct)
            avgA = sum(tTEM.vecA) / len(tTEM.vecA)
            sA = sum(tTEM.vecA)
            mA = max(tTEM.vecA)
            sT = tTEM.tStruct.shape[1]

            #TEMPs:
            bSkip = bSkip or (tTEMStruct.shape[0] > 5)
            if bSkip: continue

            if sT not in mapNumTensToSampleCount.keys(): mapNumTensToSampleCount[sT] = 1
            else: mapNumTensToSampleCount[sT] += 1
            avgT = tTEM.tStruct.shape[1] / tTEM.tStruct.shape[0]
            sTenOutDeg = sum([sum(len(vD) for vD in vvD) for vvD in tTEM.vecDestinations])
            sTenInDeg = sum([len(vD) for vD in tTEM.vecDependencies])

            vecNL = [len(d["UnaryOps"]) for d in dModelCfg["TOMData"]]
            avgNL = sum(vecNL) / len(vecNL)
            sNL = sum(vecNL)
            if sNL not in mapNumNLToSampleCount.keys(): mapNumNLToSampleCount[sNL] = 1
            else: mapNumNLToSampleCount[sNL] += 1

            vecTS = [T.cacheADFNN.GetTOM(s) for s in dModelCfg["TOMIDs"]]
            vecTS = []
            bSkip = False
            for s in dModelCfg["TOMIDs"]:
                tTOM = T.cacheADFNN.GetTOM(s)
                if tTOM is None:
                    print("WARNING: missing TOM {} from cfg {}".format(s, strID))
                    input()
                    bSkip = True
                    break
                vecTS.append(tTOM)
            if bSkip: continue
            vecTOMs = [TOMatrix(ts) for ts in vecTS]

            #arity
            vecA = [tomOp.GetArity() for tomOp in vecTOMs]
            mA = max(vecA)
            if mA not in mapAToSampleCount.keys(): mapAToSampleCount[mA] = 1
            else: mapAToSampleCount[mA] += 1
            
            #coupling arity
            vecCA = [tomOp.GetCouplingArity() for tomOp in vecTOMs]
            mCA = max(vecCA)
            if mCA not in mapCAToSampleCount.keys(): mapCAToSampleCount[mCA] = 1
            else: mapCAToSampleCount[mCA] += 1

            #order complexity
            vecOC = [tomOp.GetOrderComplexity() for tomOp in vecTOMs]
            mOC = max(vecOC)
            if mOC not in mapOCToSampleCount.keys(): mapOCToSampleCount[mOC] = 1
            else: mapOCToSampleCount[mOC] += 1

            if len(vecTOMs) not in mapNumTOMsToSampleCount.keys(): mapNumTOMsToSampleCount[len(vecTOMs)] = 1
            else: mapNumTOMsToSampleCount[len(vecTOMs)] += 1

            for i in range(len(vecTOMs)): vecTOMs[i].SetOpenModes(dModelCfg["TOMShapes"][i])
            vecContract = [t.NumContracts() for t in vecTOMs]
            vecCost = [t.ComputeCost() for t in vecTOMs]
            iTotalOps = sum([d["Multiplies"] + d["Adds"] for d in vecCost])

            #X = avgIAD + avgNL  #avgOC + sA + sT
            #X = avgContract
            #X = sT + sA + sOC
            #X = sA
            #X = sNL
            X = dResults[strID]["Params"]

            Y = dResults[strID]["TestAcc"]

            X = X.item() if isinstance(X, torch.Tensor) else X

            vecX.append(X)
            vecY.append(Y)

            strColor = vecColors[len(T.cacheADFNN.CacheMap[T.CacheMap[strID]["ADFNNModelID"]]["TOMIDs"]) - 1]
            strMarker = vecMarkers[(mA - 3) % len(vecMarkers)]
            strLabel = ax.scatter([X], [Y], s = 100, color = strColor, marker = strMarker, picker = True).get_label()
            mapLabelsToIDs[strLabel] = strID

        print("-"*42)
        print(f"Sample Count for {dCfg["ADFNNInputModule"]}-setting on {dCfg["Dataset"]}: ", len(vecX))
        print("#Ops:", mapNumTOMsToSampleCount)
        print("#Tensors:", mapNumTensToSampleCount)
        print("#NonLins:", mapNumNLToSampleCount)
        print("Arity:", mapAToSampleCount)
        print("CouplingArity:", mapCAToSampleCount)
        print("OrderComplexity:", mapOCToSampleCount)
        print("-"*42)

    ScatterTestBench()

    fig.canvas.callbacks.connect('pick_event', PickPrinter)

    plt.show()
    plt.close()

    T.cacheADFNN.Stop()

    return


def PlotParamEfficiencyCurve(dCfg: dict, strDir: str, dcParams: SearchParams, strRefDir: str = "../Trainer/") -> None:
    T = ADFNNTrainer(strDir, dCfg)

    dResults = {}
    def AddResult(strHashDir: str) -> None:
        with open(strHashDir + "Result.json", "r") as f: dResults[strHashDir.split("/")[-2]] = ProcessResult(json.load(f), dcParams)
        if not os.path.exists(strHashDir + "Model.pth"): print(f"NOTICE: No model found for {strHashDir}")
        return
    T.FindCachedStuff("Result.json", cFoundItemCB = AddResult)

    mapLabelsToIDs = {}
    def PickPrinter(mplEvent):
        strL = mplEvent.artist.get_label()
        strID = mapLabelsToIDs[strL]
        strModelID = T.CacheMap[strID]["ADFNNModelID"]
        T.cacheADFNN.PrintADFNN(strModelID)
        print("Test Acc: {:.2f}, Gen Gap: {:.2f}".format(100 * dResults[strID]["TestAcc"], 100 * dResults[strID]["GenGapAcc"]))
        print("Stacks: {}, Params: {}, Time: {}".format(T.CacheMap[strID]["ADFNNStacks"], dResults[strID]["Params"], dResults[strID]["Time"]))
        print("ADFNN ID: ", strModelID)
        print("---------------------------------")
        
        return
    
    fig, ax = plt.subplots(figsize=(10, 7.25))

    iMaxX = 0
    vecX = []
    vecY = []
    for strID in dResults.keys():
        #filter based on dataset and architectural setting
        if T.CacheMap[strID]["Dataset"] != dCfg["Dataset"]: continue

        if T.CacheMap[strID]["Model"] != "ADFNN": continue
        if T.CacheMap[strID]["ADFNNInputModule"] != dCfg["ADFNNInputModule"]: continue

        #filter based on training recipe
        if T.CacheMap[strID]["NumEpochs"] != dCfg["NumEpochs"]: continue
        if T.CacheMap[strID]["DataAugmentation"] != dCfg["DataAugmentation"]: continue
        if T.CacheMap[strID]["LRScheduler"] != dCfg["LRScheduler"]: continue
        if T.CacheMap[strID]["SubSample"] != dCfg["SubSample"]: continue

        #throw out the garbage
        if len(dResults[strID]["Runs"][0]["TestMetrics"]) <= dcParams.iGiveUpE: continue

        #grab the ADFNN info
        strADFNNID = T.CacheMap[strID]["ADFNNModelID"]
        dModelCfg = T.cacheADFNN.CacheMap[strADFNNID]

        #ensure we haven't lost any data due to git weirdness
        tTEMStruct = T.cacheADFNN.GetTEM(dModelCfg["TEMID"])
        if tTEMStruct is None:
            print("WARNING: missing TEM {} from cfg {}".format(dModelCfg["TEMID"], strID))
            input()
            continue
        tTEM = TEMatrix(tTEMStruct)
        mA = max(tTEM.vecA)

        vecTS = [T.cacheADFNN.GetTOM(s) for s in dModelCfg["TOMIDs"]]
        vecTS = []
        bSkip = False
        for s in dModelCfg["TOMIDs"]:
            tTOM = T.cacheADFNN.GetTOM(s)
            if tTOM is None:
                print("WARNING: missing TOM {} from cfg {}".format(s, strID))
                input()
                bSkip = True
                break
            vecTS.append(tTOM)
        if bSkip: continue

        #add the data point
        X = dResults[strID]["Params"]
        if X > iMaxX: iMaxX = X
        Y = 100*dResults[strID]["TestAcc"]

        strMarker = "o"
        strColor = "deepskyblue"
        strLabel = ax.scatter([X], [Y], s = 75, color = strColor, marker = strMarker, picker = True).get_label()
        mapLabelsToIDs[strLabel] = strID


    #Collect the reference points
    for strK in REFERENCE_MODELS.keys():
        if strK in dCfg["ADFNNInputModule"]: vecModels = REFERENCE_MODELS[strK]

    vecX = []
    vecY = []
    vecStd = []
    TRef = ADFNNTrainer(strRefDir, dCfg)
    print(vecModels)
    for strM in vecModels:
        TRef.UpdateModel(strM)
        for strK in LEARNING_RATES.keys():
            if strK in TRef.dCfg["Model"]:
                TRef.dCfg["LearningRate"] = LEARNING_RATES[strK]
        TRef.dCfg["NumRuns"] = 3 #make sure we get error bars for the "context" architectures
        print(TRef.IGenHash(bPrint = True))
        TRef.UpdateCacheMap()
        dR = ProcessResult(TRef.GetResult(bY = True), dcParams)
        X = dR["Params"]
        if X > iMaxX: iMaxX = X
        vecX.append(X)
        vecY.append(100*dR["AvgTestAcc"])
        vecStd.append(100*dR["StdTestAcc"])

    print(vecX, vecY, vecStd)
    #ax.plot(vecX, vecY, linewidth = 5, marker = "s", markersize = 12, color = "black")
    ax.errorbar(vecX, vecY, yerr=vecStd, linewidth = 5, marker = "s", markersize = 12, color = "black")

    #plot the stage1 baseline
    strM = MAP_INPUTMODULES_TO_STEMS[dCfg["ADFNNInputModule"]]
    TRef.UpdateModel(strM)
    TRef.dCfg["NumRuns"] = 3 #make sure we get error bars for the "context" architectures
    TRef.UpdateCacheMap()
    dR = ProcessResult(TRef.GetResult(bY = True), dcParams)
    Y = 100*dR["AvgTestAcc"]
    ax.plot([0, iMaxX], [Y, Y], linewidth = 5, linestyle = "dashed", color = "black")

    ax.set_xlabel("Number of Parameters", fontsize = 26)
    ax.semilogx()
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize = 26)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    ax.grid(linewidth = 1.15, linestyle = "dashed", color = "darkgrey", which="both")

    ax.set_title(vecModels[0] + ": " + dCfg["Dataset"], fontsize = 28)

    fig.canvas.callbacks.connect('pick_event', PickPrinter)

    plt.show()
    plt.close()

    T.cacheADFNN.Stop()
    TRef.cacheADFNN.Stop()

    return


def ScatterVsX(strDatasetDir: str, vecDS: list[str], vecArch: list[str], dcParams: SearchParams, strRefDir: str = "../Trainer/", strX: str = "Params") -> None:

    assert strX in ["Params", "OpC", "TC", "AC", "OC", "CAC", "NL"], "Invalid strX: {}".format(strX)

    mapXToDisplay = {"Params": "Number of Parameters",
                     "OpC": "Operation Complexity",
                     "TC": "Tensor Complexity",
                     "AC": "Arity Complexity",
                     "OC": "Order Complexity",
                     "CAC": "Coupling Arity Complexity",
                     "NL": "Number of Non-Linearities"
                    }

    def AddResult(strHashDir: str) -> None:
        with open(strHashDir + "Result.json", "r") as f: dResults[strHashDir.split("/")[-2]] = ProcessResult(json.load(f), dcParams)
        if not os.path.exists(strHashDir + "Model.pth"): print(f"NOTICE: No model found for {strHashDir}")
        return
    
    vecColors = ["midnightblue", "dodgerblue", "skyblue", "maroon", "orange", "darkcyan", "green", "orchid"]
    vecMarkers = ["v", "s", "*"]

    fig, axes = plt.subplots(nrows = len(vecArch), ncols = len(vecDS), figsize=(14, 9))
    #mpl does not handle trivial array dims nicely
    if len(vecArch) == 1: axes = [axes]
    if len(vecDS) == 1:
        for i in range(len(axes)): axes[i] = [axes[i]]

    for i in range(len(vecArch)): axes[i][0].set_ylabel(vecArch[i], fontsize = 20)
    
    for j in range(len(vecDS)):
        strDS = vecDS[j]
        dCfg["Dataset"] = strDS
        axes[0][j].set_title(strDS, fontsize = 20)
        for i in range(len(vecArch)):
            strArch = vecArch[i]

            ax = axes[i][j]

            strDir = strDatasetDir + strDS + "/" + strArch + "/"
            dCfg["ADFNNInputModule"] = mapArchToInputModule[strArch]

            T = ADFNNTrainer(strDir, dCfg)

            dResults = {}
            T.FindCachedStuff("Result.json", cFoundItemCB = AddResult)

            mapLabelsToIDs = {}
            def PickPrinter(mplEvent):
                strL = mplEvent.artist.get_label()
                strID = mapLabelsToIDs[strL]
                strModelID = T.CacheMap[strID]["ADFNNModelID"]
                T.cacheADFNN.PrintADFNN(strModelID)
                print("Test Acc: {:.2f}, Gen Gap: {:.2f}".format(100 * dResults[strID]["TestAcc"], 100 * dResults[strID]["GenGapAcc"]))
                print("Stacks: {}, Params: {}, Time: {}".format(T.CacheMap[strID]["ADFNNStacks"], dResults[strID]["Params"], dResults[strID]["Time"]))
                print("ADFNN ID: ", strModelID)
                print("---------------------------------")
                
                return


            iMaxX = 0
            mapXToYs = {}
            vecX = []
            vecY = []
            for strID in dResults.keys():
                #filter based on dataset and architectural setting
                if T.CacheMap[strID]["Dataset"] != dCfg["Dataset"]: continue

                if T.CacheMap[strID]["Model"] != "ADFNN": continue
                if T.CacheMap[strID]["ADFNNInputModule"] != dCfg["ADFNNInputModule"]: continue

                #filter based on training recipe
                if T.CacheMap[strID]["NumEpochs"] != dCfg["NumEpochs"]: continue
                if T.CacheMap[strID]["DataAugmentation"] != dCfg["DataAugmentation"]: continue
                if T.CacheMap[strID]["LRScheduler"] != dCfg["LRScheduler"]: continue
                if T.CacheMap[strID]["SubSample"] != dCfg["SubSample"]: continue

                #throw out the garbage
                if len(dResults[strID]["Runs"][0]["TestMetrics"]) <= dcParams.iGiveUpE: continue

                #grab the ADFNN info
                strADFNNID = T.CacheMap[strID]["ADFNNModelID"]
                dModelCfg = T.cacheADFNN.CacheMap[strADFNNID]

                #ensure we haven't lost any data due to git weirdness
                tTEMStruct = T.cacheADFNN.GetTEM(dModelCfg["TEMID"])
                if tTEMStruct is None:
                    print("WARNING: missing TEM {} from cfg {}".format(dModelCfg["TEMID"], strID))
                    input()
                    continue
                tTEM = TEMatrix(tTEMStruct)
                mA = max(tTEM.vecA).item()

                vecTS = [T.cacheADFNN.GetTOM(s) for s in dModelCfg["TOMIDs"]]
                vecTS = []
                bSkip = False
                mOC = 1
                mCA = 1
                for s in dModelCfg["TOMIDs"]:
                    tTOM = T.cacheADFNN.GetTOM(s)
                    if tTOM is None:
                        print("WARNING: missing TOM {} from cfg {}".format(s, strID))
                        input()
                        bSkip = True
                        break
                    if tTOM.shape[1] > mOC: mOC = tTOM.shape[1]
                    vecTS.append(TOMatrix(tTOM))
                    iCA = vecTS[-1].GetCouplingArity()
                    if iCA > mCA: mCA = iCA
                
                vecNL = [len(d["UnaryOps"]) for d in dModelCfg["TOMData"]]
                sNL = sum(vecNL)
                
                if bSkip: continue

                #add the data point
                if strX == "Params": X = dResults[strID]["Params"]
                elif strX == "OpC": X = tTEM.tStruct.shape[0]
                elif strX == "TC": X = tTEM.tStruct.shape[1]
                elif strX == "AC": X = mA
                elif strX == "OC": X = mOC
                elif strX == "CAC": X = mCA
                elif strX == "NL": X = sNL

                if X > iMaxX: iMaxX = X
                Y = 100*dResults[strID]["TestAcc"]

                vecX.append(X)
                vecY.append(Y)

                if X not in mapXToYs.keys(): mapXToYs[X] = []
                mapXToYs[X].append(Y)

            strMarker = "o"
            strColor = "deepskyblue"
            ax.scatter(vecX, vecY, s = 75, color = strColor, marker = strMarker, picker = False).get_label()

            #plot the stage1 baseline
            TRef = ADFNNTrainer(strRefDir, dCfg)
            strM = MAP_INPUTMODULES_TO_STEMS[dCfg["ADFNNInputModule"]]
            TRef.UpdateModel(strM)
            for strK in LEARNING_RATES.keys():
                if strK in TRef.dCfg["Model"]:
                    TRef.dCfg["LearningRate"] = LEARNING_RATES[strK]
            TRef.dCfg["NumRuns"] = 3 #make sure we get error bars for the "context" architectures
            TRef.UpdateCacheMap()
            dR = ProcessResult(TRef.GetResult(bY = True), dcParams)
            Y = 100*dR["AvgTestAcc"]
            ax.plot([0, iMaxX], [Y, Y], linewidth = 3, linestyle = "solid", color = "black")

            if strX == "Params": ax.semilogx()
            
            ax.grid(linewidth = 1.15, linestyle = "dashed", color = "darkgrey", which="both")
            ax.tick_params(axis='both', which='major', labelsize=18)
            
            if strX != "Params":
                vecXMed = sorted([strK for strK in mapXToYs.keys()])
                vecYMed = [torch.median(torch.tensor(mapXToYs[strK])) for strK in vecXMed]
                vecYMean = [torch.mean(torch.tensor(mapXToYs[strK])) for strK in vecXMed]
                ax.plot(vecXMed, vecYMed, linewidth = 3, linestyle = "dashed", color = "black")
                ax.plot(vecXMed, vecYMean, linewidth = 3, linestyle = "dashed", color = "orange")

            #ax.set_title(vecModels[0] + ": " + dCfg["Dataset"], fontsize = 28)
            #fig.canvas.callbacks.connect('pick_event', PickPrinter)
    
    fig.suptitle(mapXToDisplay[strX] + " vs. Accuracy", fontsize = 24)

    plt.show()
    plt.close()

    T.cacheADFNN.Stop()
    TRef.cacheADFNN.Stop()

    return


def PlotDatasetCollection(strDatasetDir: str, vecDS: list[str], vecArch: list[str], dcParams: SearchParams) -> None:
    for strDS in vecDS:
        dCfg["Dataset"] = strDS
        for strArch in vecArch:
            strDir = strDatasetDir + strDS + "/" + strArch + "/"
            dCfg["ADFNNInputModule"] = mapArchToInputModule[strArch]
            PlotADFNNSearch(dCfg = dCfg, strDir = strDir, dcParams = dcParams)

    return

def PlotPvsA(strDatasetDir: str, vecDS: list[str], vecArch: list[str], dcParams: SearchParams) -> None:
    for strDS in vecDS:
        dCfg["Dataset"] = strDS
        for strArch in vecArch:
            strDir = strDatasetDir + strDS + "/" + strArch + "/"
            dCfg["ADFNNInputModule"] = mapArchToInputModule[strArch]
            PlotParamEfficiencyCurve(dCfg = dCfg, strDir = strDir, dcParams = dcParams)

    return


if __name__ == "__main__":

    import sys

    vecDS = ["CIFAR10", "CIFAR100F", "TinyImagenet"]
    vecArch = ["ResNet34H", "SWIN_T"]

    bAM = False
    bPA = False
    strDS = None
    strArch = None
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-m": strArch = sys.argv[i+1]
        if sys.argv[i] == "-d": strDS = sys.argv[i+1]
        if sys.argv[i] == "-PA": bPA = True
        if sys.argv[i] == "-x":
            bAM = True
            strX = sys.argv[i+1]

    if strDS:
        for strK in vecDS:
            if strDS.lower() in strK.lower():
                strDS = strK
                break
        vecDS = [strDS]
    if strArch:
        for strK in vecArch:
            if strArch.lower() in strK.lower():
                strArch = strK
                break
        vecArch = [strArch]

    mapArchToInputModule = {
        "ResNet34H": "ResNet34HStage1",
        "SWIN_T": "SWIN_TStage1",
    }

    dCfg = {
        "Model": "ADFNN",
        "ADFNNCacheDir": "ADFNNCache",

        "ADFNNInputModule": mapArchToInputModule[strArch] if strArch else "",

        "Dataset": strDS if strDS else "",
        "DataAugmentation": True,
        "LRScheduler": "OneCycle",
        "BatchSize": 128,
        "SubSample": 1.0,
        "NumEpochs": 50,
    }

    dcParams = SearchParams(iTOMsPerModel = 5,
                            fUnaryOpProbability = 0.5,
                            iGiveUpE = 10)

    strDatasetDir = "../Dataset/"

    if bPA: PlotPvsA(strDatasetDir, vecDS, vecArch, dcParams)
    elif bAM: ScatterVsX(strDatasetDir, vecDS, vecArch, dcParams, strX = strX)
    else: PlotDatasetCollection(strDatasetDir, vecDS, vecArch, dcParams)