from sklearn.metrics import f1_score,accuracy_score,recall_score,\
precision_score,confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,precision_recall_curve,auc
import os
from itertools import cycle
from pandas import read_csv
from matplotlib import pyplot as plt
import numpy as np
from numpy import interp
from sklearn.utils.multiclass import type_of_target

'''本程序需要先调用计算pr和roc的函数，再绘制pr和roc曲线'''


def scores(y_test,y_pred,th=0.5,lower = False):
    '''
功能：计算评价指标Recall,SPE,Precision,F1,MCC,Acc,AUC,aupr,tp,fn,tn,fp
输入参数：
    y_test:测试样本的标签
    y_pred: 方法对测试样本的预测值
    th: 分类阈值，默认为0.5
输出：
    评价指标列表：Recall,SPE,Precision,F1,MCC,Acc,AUC,aupr,tp,fn,tn,fp
'''
    if lower == False:
        y_predlabel=[(0 if item<th else 1) for item in y_pred]
    elif lower == True:
        y_predlabel = [(0 if item > th else 1) for item in y_pred]

    tn,fp,fn,tp=confusion_matrix(y_test,y_predlabel).flatten()
    SPE=tn*1./(tn+fp)
    MCC=matthews_corrcoef(y_test,y_predlabel)
    Recall=recall_score(y_test, y_predlabel)
    Precision=precision_score(y_test, y_predlabel)
    F1=f1_score(y_test, y_predlabel)
    Acc=accuracy_score(y_test, y_predlabel)
    AUC=roc_auc_score(y_test, y_pred)
    pr, rc, _ = precision_recall_curve(y_test, y_pred)
    aupr = auc(rc, pr)
    return [Recall,SPE,Precision,F1,MCC,Acc,AUC,aupr,tp,fn,tn,fp]



def draw_single_ROC(predfile='',data=[],header=True,saved=False,fontsize=16):
    '''
    绘制单个预测结果文件的ROC曲线。
    参数：
    predfile：预测结果的文件名，默认无。文件可接受TXT和CSV格式。内容由两列组成，第一列为标签，第二列为预测值；
    header:配合predfile使用的，默认为预测结果含有标题行。
    data：预测结果的列表，第一个列表元素为标签值，第二个列表元素为预测值；
    saved：给出是否保存还是直接显示，默认为直接显示；
    fontsize为字号，默认为16号
'''

    if predfile != '':
        print(os.path.splitext(predfile)[-1])
        if os.path.splitext(predfile)[-1] == '.csv':
            if header == False:
                results = read_csv(predfile,header=None).dropna()
            else:
                results = read_csv(predfile).dropna()
        elif os.path.splitext(predfile)[-1] == '.txt':
            if header == False:
                results = read_csv(predfile, sep='\t',header=None).dropna()
            else:
                results = read_csv(predfile,sep='\t').dropna()
        else:
            print('仅接受txt或者csv格式文件。。。')
            os.exit()

        print(results.head())

        #计算roc曲线对应的每组值
        fpr,tpr,_ = roc_curve(results.iloc[:,0].values,results.iloc[:,1].values)#第一列为标签，第二列为预测值
        auc_stat = roc_auc_score(results.iloc[:,0].values,results.iloc[:,1].values)

    if len(data) != 0:
        fpr, tpr, _ = roc_curve(data[0], data[1])  # 第一个列表元素存放标签，第二个列表元素存放预测值
        auc_stat = roc_auc_score(data[0], data[1])

    fig = plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')

    plt.plot(fpr, tpr, linestyle='-', lw=2,  label='(AUC = {:.2f})'.format( auc_stat))
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    # plt.title('ROC curve',fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=fontsize)
    if saved == True:
        filename = os.path.splitext(predfile)[0] + '.tiff'
        fig.savefig(filename, format='tiff', dpi=300,bbox_inches='tight',pil_kwargs={'compression': 'tiff_lzw'})
        plt.close()
    else:
        plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

def cal_rocs(results):
    '''
    计算文件中各个方法的roc相关指标，并返回结果用于后续调用print_ROC_figure
    results参数：该DataFrame第一列为标签，第二列及以后列为各方法的预测值。
    返回：fprs, tprs, auc_stats,methodnames
    '''
    
    fprs = []
    tprs = []
    aurocs = []
    for i in range(1,len(results.columns)):
        # 计算roc曲线对应的每组值
        fpr, tpr, _ = roc_curve(results.iloc[:, 0].values, results.iloc[:, i].values)  # 第一列为标签，第二列为预测值
        auc_stat = roc_auc_score(results.iloc[:, 0].values, results.iloc[:, i].values)
        fprs.append(fpr)
        tprs.append(tpr)
        aurocs.append(auc_stat)
    return fprs,tprs,aurocs,results.columns[1:]


def print_ROC_figure(filename, fprs, tprs, auc_stats,methodnames, fontsize=14,aucsorted =False):
    '''
    功能：绘制多个方法的roc曲线，并保存成.png格式的文件
    输入参数：
        filename:待保存的文件位置和文件名
        fprs: 存放多个fpr的列表
        tprs: 存放多个tpr的列表
        methodnames: 存放所比较的方法的名称列表
    输出：
        ROC曲线以.png格式和filename存放。
    '''    
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    
    if len(methodnames) >= 9:
        colors = ['#FF7F00','#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#FDBF6F',  '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928']# Paired,去掉红色，剩11种颜色
    else:
        colors= ["#FC8D62", "#66C2A5", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494","#B3B3B3"]# Set2 有8种颜色
    
    
    linestyles = cycle(['-','-.','--',':'])
    markers = cycle(["o", "+", "^", "<", "x", "s", "p", "*", "H", "h", "D"])
    # fig = plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')

    if aucsorted==True:
        fprs, tprs, auc_stats, methodnames=auc_sort(fprs, tprs, auc_stats, methodnames)
    for methodname, fpr,tpr,auc_stat,color,linestyle,marker in zip(methodnames,fprs,tprs,auc_stats,colors,linestyles,markers):
        plt.plot(fpr, tpr, linestyle=linestyle, lw=1, color=color,marker=marker,markersize=5,markevery=20,label='{} (AUC = {:.2f})'.format(methodname,auc_stat))
       
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('False Positive Rate',fontsize=fontsize)
    plt.ylabel('True Positive Rate',fontsize=fontsize)
    # plt.title('ROC curve',fontsize=fontsize)
    plt.legend(loc="lower right",fontsize=fontsize-2)
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')#,pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)

def cal_prs(results):
    '''
    计算文件中各个方法的pr相关指标，并返回结果用于后续调用print_PR_figure
    results参数：该DataFrame第一列为标签，第二列及以后列为各方法的预测值。
    返回：recs, pres, auprs,methodnames
    '''
    
    recs = []
    pres = []
    auprs = []
    for i in range(1,len(results.columns)):
        # 计算roc曲线对应的每组值
        pre,rec, _ = precision_recall_curve(results.iloc[:, 0].values, results.iloc[:, i].values)  # 第一列为标签，第二列为预测值
        aupr_stat = auc(rec, pre)
        recs.append(rec)
        pres.append(pre)
        auprs.append(aupr_stat)
    return recs, pres,auprs,results.columns[1:]

def print_PR_figure(filename, rcs, prs,auc_stats,methodnames, fontsize=14,aucsorted =False):

    '''
    功能：绘制多个方法的PR曲线，并保存成.png格式的文件
    输入参数：
        filename:待保存的文件位置和文件名    
        rcs: 存放多个recall的列表
        prs: 存放多个precisions的列表
        methodnames: 存放所比较的方法的名称列表
    输出：
        PR曲线以.png格式和filename存放。
    '''
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    # colors = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928']# Paired,12种颜色
    if len(methodnames) >= 9:
        colors = ['#FF7F00','#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#FDBF6F',  '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928']# Paired,去掉红色，剩11种颜色
    else:
        colors= ["#FC8D62","#66C2A5",  "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494","#B3B3B3"]# Set2 有8种颜色
    # colors = ['#8E0152', '#C51B7D', '#DE77AE', '#F1B6DA', '#FDE0EF', '#F7F7F7', '#E6F5D0', '#B8E186', '#7FBC41', '#4D9221', '#276419'] #'PiYG',11种颜色

    # 
    # colors = [ "#543005", "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3", "#F5F5F5", "#C7EAE5","#80CDC1", "#35978F", "#01665E", "#003C30"]  # 'BrBG'有11种颜色
  
    linestyles = cycle(['-','-.','--',':'])
    markers = cycle(["o", "+", "^", "<", "x", "s", "p", "*", "H", "h", "D"])
    # fig = plt.figure(figsize=(8, 8))
    # plt.plot([0, 1], [1, 0], linestyle='--', lw=1, color='k')
    if aucsorted==True:
        rcs, prs,auc_stats,methodnames=auc_sort(rcs, prs,auc_stats,methodnames)
    for methodname, pr,rc,auc_stat,color,linestyle,marker in zip(methodnames,prs,rcs,auc_stats,colors,linestyles,markers):
        
        plt.plot(rc[::], pr[::],  linestyle=linestyle,marker=marker,markersize=3, lw=1, markevery=20, color=color,label='{} (AUPR = {:.2f})'.format(methodname,auc_stat))  #::-1
        

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Recall',fontsize=fontsize)
    plt.ylabel('Precision',fontsize=fontsize)
    # plt.title('PR curve',fontsize=fontsize)
    plt.legend(loc="lower right",fontsize=fontsize-2)
    plt.savefig(filename, format='png', dpi=300,bbox_inches='tight')#,pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def print_ROC_PR_figure(filename, fprs, tprs, auc_stats,methodnames_roc, rcs, prs,aupr_stats,methodnames_pr, fontsize=14):

    '''
    功能：同时绘制多个方法的ROC和PR曲线，并保存成.png格式的文件。
    输入参数：
        filename:待保存的文件位置和文件名
        fprs: 存放多个fpr的列表
        tprs: 存放多个tpr的列表
        methodnames_roc: 存放roc所比较的方法的名称列表
        rcs: 存放多个recall的列表
        prs: 存放多个precisions的列表
        methodnames_pr: 存放pr所比较的方法的名称列表
    说明：因为roc曲线和pr曲线数据可能先通过auc和aupr进行过排序，因此对应的方法名称排序也会不同，因此分别输入之。
    输出：
        左图ROC曲线，右图PR曲线。以.png格式和filename存放。
    '''
    assert isinstance(filename, str), 'filename must be a string'
    filename = os.path.splitext(filename)[0] + '.png'

    colors = cycle([ "#CC79A7","#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00","#0000FF","#6666FF","#000000"])

    fig = plt.figure(figsize=(17, 8))

    ax1 = fig.add_subplot(121)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')
    for methodname, fpr,tpr,auc_stat,color in zip(methodnames_roc,fprs,tprs,auc_stats,colors):
        ax1.plot(fpr, tpr,  linestyle='-',  lw=2, color=color,label='{} (AUC = {:.3f})'.format(methodname,auc_stat))
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate',fontsize=fontsize)
    ax1.set_ylabel('True Positive Rate',fontsize=fontsize)
    # ax1.set_title('ROC curve',fontsize=fontsize)
    ax1.legend(loc="lower right",fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)

    ax2 = fig.add_subplot(122)
    # plt.plot([0, 1], [1, 0], linestyle='--', lw=1, color='k')
    for methodname, pr, rc, aupr_stat, color in zip(methodnames_pr, prs, rcs, aupr_stats, colors):
        ax2.plot(rc[::-1], pr[::-1], linestyle='-', lw=2, color=color,
                 label='{} (AUPR = {:.3f})'.format(methodname, aupr_stat))
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall', fontsize=fontsize)
    ax2.set_ylabel('Precision', fontsize=fontsize)
    # ax2.set_title('PR curve', fontsize=fontsize)
    ax2.legend(loc="lower right", fontsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    fig.savefig(filename, format='png',dpi=300, bbox_inches='tight')

    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def auc_sort(fprs, tprs, aucs, methodnames):

   '''
    功能：对AUC/AUPR值进行排序,以便再后续绘制曲线时的图例中按照AUC/AUPR值进行降序显示。
    输入参数：
        fprs: 存放多个fpr/recall的列表
        tprs: 存放多个tpr/precision的列表
        aucs: 存放多个auc/aupr的列表
        methodnames: 存放所比较的方法的名称列表    
    输出：
        按照AUC值作为排序依据后的对应排序后fprs,tprs,aucs,methodnames
    '''
   auc_desend = sorted(range(len(aucs)), key=lambda k: aucs[k], reverse=True)
   fprs_sorted = []
   tprs_sorted = []
   aucs_sorted = []
   methodnames_sorted = []
   for i in auc_desend:
        tprs_sorted.append(tprs[i])
        fprs_sorted.append(fprs[i])
        aucs_sorted.append(aucs[i])
        methodnames_sorted.append(methodnames[i])
   return fprs_sorted, tprs_sorted, aucs_sorted,methodnames_sorted






def draw_kfold_roc_pr(clf_names,k,fontsize = 16,saved=False,filepath='d:\\temp\\'):

    '''
    功能：绘制多种机器学习方法在k折交叉验证上的pr曲线和roc曲线
    输入参数：
        clf_names:机器学习方法的名称，应该与之前保存的K折交叉验证结果保持一致，以方便在方法中通过循环读取每一个文件内容； 
        k: k折，一般为10折； 
        fontsize:绘制图形中的字号大小，默认为16. 
        saved:是否保存绘制曲线结果，默认不保存，直接显示；
        filepath:保存曲线的路径，默认为'd:\\temp\\'.
            
    输出：
        无。
        
    '''
    colors = ['red', 'blue', 'darkorange', 'seagreen', 'gold', 'orchid', 'sienna', 'gray', 'cyan', 'black']
    fig = plt.figure(figsize=(17, 8))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k')

    for i,clf_name in enumerate(clf_names):

        #此处装载文件的名称需要在调用时根据实际情况调整。
        output10fold = np.load(r'../results/Train_10fold_proba_{}1213_top10.npy'.format(clf_name))#数据前10列为十折预测结果，后10列为对应标签。

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0,1,100)

        precs = []
        auprs = []
        mean_recall = np.linspace(0, 1, 100)

        for j in range(k):
            # 计算roc曲线和auc
            fpr,tpr,threshholds = roc_curve(output10fold[:,10+j],output10fold[:,j])
            tprs.append(interp(mean_fpr,fpr,tpr))# mean_fpr为待插值点的横坐标，fpr,tpr分别为已知点的横纵坐标
            tprs[-1][0]= 0.0  #使用-1获得列表的最后一个元素，即取当前新加入的插值向量中的第一个值
            roc_auc = auc(fpr,tpr)
            aucs.append(roc_auc)

            # 计算pr曲线和aupr
            prec, recall, _ = precision_recall_curve(output10fold[:,10+j],output10fold[:,j])
            print(interp(mean_recall,recall,prec))
            precs.append(interp(mean_recall,recall[::-1],prec[::-1]))
            precs[-1][-1]=0.0

            auprs.append(auc(recall, prec))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
        mean_auc = auc(mean_fpr, mean_tpr)

        ax1.plot(mean_fpr, mean_tpr, linestyle='-', lw=2, color=colors[i], label='{} (AUC = {:.3f})'.format(clf_name, mean_auc))

        mean_prec = np.mean(precs, axis=0)
        mean_prec[-1] = 0.0  # 坐标最后一个点为（1,0）  以0为终点
        mean_aupr = auc(mean_recall, mean_prec)


        ax2.plot(mean_recall[::-1], mean_prec[::-1], linestyle='-', lw=2, color=colors[i],label='{} (AUPR = {:.3f})'.format(clf_name, mean_aupr))

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    ax1.set_xlabel('False Positive Rate', fontsize=fontsize)
    ax1.set_ylabel('True Positive Rate', fontsize=fontsize)
    # ax1.title('ROC curve',fontsize=fontsize)
    ax1.legend(loc="lower right", fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    # ax2.set_xticks()
    # ax2.set_yticks()
    ax2.set_xlabel('Recall', fontsize=fontsize)
    ax2.set_ylabel('Precision', fontsize=fontsize)
    # ax2.title('ROC curve',fontsize=fontsize)
    ax2.legend(loc="lower right", fontsize=fontsize)
    # 设置坐标刻度值的大小以及刻度值的字体
    ax2.tick_params(labelsize=fontsize)
    if saved == True:
        filename = f'{filepath}{k}fold_compare.png'
        fig.savefig(filename, format='png', dpi=300,bbox_inches='tight')
        plt.close()
    else:
        plt.show()



if __name__ == '__main__':

################### 测试函数功能是否正常######################################

    #1. 测试绘制单个roc曲线，通过文件传入预测结果和标签
    # draw_single_ROC(predfile=r'F:\CMMPred\20201018\data\CHASM_indep_scores1058.txt')

    # 方法1：通过读取文件获得标签和预测值
    # draw_single_ROC(predfile=r'F:\CMMPred\20201018\results\cscape_somatic_1058_1.txt')
    # 方法2：通过列表数据传入，列表第一个元素为所有的标签值；第二个元素为对应的所有预测值。
    data = read_csv(r'F:\CMMPred\20201018\results\cscape_somatic_1058_1.txt',sep='\t').dropna().values
    data = data.T.tolist()
    draw_single_ROC(data=data)

    # 2. 测试绘制单个roc曲线，通过列表元素直接传入预测结果和标签
    # output10fold = np.load(r'../results/Train_10fold_proba_{}1213_top10.npy'.format('XGB'))
    # list_label = output10fold[:,10]
    # list_preds = output10fold[:,0]
    # data = [list_label,list_preds]
    # draw_single_ROC(data=data)


    # 3. 绘制多个方法在K折交叉验证上的ROC和PR曲线
    # clf_names = ['XGB', 'SVM', 'MLP', 'RF', 'GaussianNB', 'LR', 'LDA', 'GBDT']
    # # clf_names = ['SVM']
    # draw_kfold_roc_pr(clf_names,10)#直接显示曲线
    # draw_kfold_roc_pr(clf_names,10,saved=True)#保存曲线到默认路径，可通过给filepath传入其它参数，修改保存路径。

'''
功能：
输入参数：
    : 
    : 
    : 

输出：

'''