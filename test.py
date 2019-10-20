import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data=[]
    labels=[] 
    with open(filename,'r') as f:
        for line in f:
            xz,yz,label = [float(i) for i in line.strip().split()] ## 去掉回车 并且按照空格进行分隔
            data.append([xz,yz])
            labels.append(label)
    return data,labels
def clip(alpha ,L,H):
    """
        进行剪枝操作
    """
    if alpha < L:
        return L
    elif alpha > H:
        return H
    else:
        return alpha
def select_j(i,m):
    """
        在m个数里面随机选择一个j
    """
    L = list(range(m))
    L = L[:i]+L[i+1:]
    return np.random.choice(L)
def get_w(alpha,dataSet,labels):
    """
        根据alpha ,dataSet,labels 获取超平面参数 w
    """
    alpha,dataSet,labels = np.array(alpha) ,np.array(dataSet),np.array(labels)
    yx = labels.reshape(1,-1).T *np.array([1,1]) * dataSet 
    w = np.dot(yx.T,alpha)
    return w.tolist()
def simple_smo(dataSet,labels,max_iter,C):
    """
        简化版smo算法 
        dataSet 就是用于训练的数据集(m,n) m代表样本数量 n代表特征数量
        labels 就是其类别 +1 or -1
        C代表权衡因子 就是目标函数里面权衡 惩罚项 与 间隔的之间的比例系数 
            当C越大的时候 误分的数量就越少 但此时不一定保证间隔最大
            当C越小的时候 间隔越大 抗干扰能力越好
        max_iter 指最大迭代次数 如果在max_iter步没有进行优化alpha 则认为优化结束
    """
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    m,n = dataSet.shape
    ## 初始化参数
    alpha = np.zeros(m)
    b = 0
    def f(x):
        """
            分类超平面
            f(x) = w.T * x + b
        """
          # Kernel function vector.
        x = np.matrix(x).T
        data = np.matrix(dataSet)
        ks = data*x
        # Predictive value.
        wx = np.matrix(alpha*labels)*ks
        fx = wx + b
        return fx[0, 0]
    it = 0
    while it < max_iter:
        piror_step =0
        for i in range(m):
            a_i,x_i,y_i = alpha[i],dataSet[i],labels[i]
            f_i = f(x_i)
            E_i = f_i - y_i
            j = select_j(i,m)
            a_j ,x_j,y_j = alpha[j],dataSet[j],labels[j]
            f_j = f(x_j)
            E_j = f_j - y_j
            k_ii,k_ij,k_jj = np.dot(x_i,x_i),np.dot(x_i,x_j),np.dot(x_j,x_j)
            eta = k_ii +k_jj - 2*k_ij
            if eta <=0:
                print("Error")
                continue
            a_i_old,a_j_old = a_i,a_j
            a_j_new = a_j_old + y_j *(E_i-E_j)/eta
            if y_i != y_j:
                L = max(0,a_j_old - a_i_old)
                H = min(C,C+a_j_old - a_i_old)
            else:
                L = max(0,a_j_old+a_i_old-C)
                H = min(C,a_i_old+a_j_old)
            a_j_new = clip(a_j_new,L,H)
            a_i_new = a_i_old + y_i * y_j *(a_j_old - a_j_new)
            if abs(a_j_new - a_j_old) < 1e-6:
                continue
            alpha[i] = a_i_new
            alpha[j] = a_j_new
            ## 更新阈值b
            b_i = -E_i - y_i * k_ii*(a_i_new - a_i_old) - y_j * k_ij*(a_j_new - a_j_old) + b
            b_j = -E_j - y_i * k_ij*(a_i_new - a_i_old) - y_j * k_jj*(a_j_new - a_j_old) + b
            if 0 < a_i_new < C:
                b = b_i
            elif 0 < a_j_new < C:
                b = b_j
            else:
                b = (b_i + b_j)/2
            piror_step +=1
            print('INFO itearation:{0} i:{1} piror_step:{2}'.format(it,i,piror_step))
        if piror_step == 0:
            it +=1
        else:
            it = 0
        print('iteration number: {}'.format(it))
    return alpha, b
if __name__ == "__main__":
    dataSet,labels = load_data("dataSet/SVMdataSet/trainSet.txt")
    alpha , b = simple_smo(dataSet,labels,40,0.6)
    print(alpha)
    ## 分类数据点
    classified = {'+1':[],'-1':[]}
    for point ,label in zip(dataSet,labels):
        if label == 1.0 :
            classified['+1'].append(point)
        else :
            classified['-1'].append(point)
    ## 绘制数据点
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ## 绘制数据点
    for label,pts in classified.items():
        pts = np.array(pts)
        ax.scatter(pts[:,0],pts[:,1],label=label)
    w = get_w(alpha, dataSet, labels)
    x1, _ = max(dataSet, key=lambda x: x[0])
    x2, _ = min(dataSet, key=lambda x: x[0])
    a1, a2 = w
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    ax.plot([x1, x2], [y1, y2])
    # 绘制支持向量
    for i, alpha in enumerate(alpha):
        if abs(alpha) > 1e-3:
            x, y = dataSet[i]
            ax.scatter([x], [y], s=150, c='none', alpha=0.7,
                       linewidth=1.5, edgecolor='#AB3319')
    plt.show()

