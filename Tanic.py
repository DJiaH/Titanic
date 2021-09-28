#PyTorch
import torch
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('display.max_rows',None)
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import seaborn as sns  #matplotlib的基础上封装的库，方便直接传参调用,进行数据可视化
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import csv
import os



train=pd.read_csv('./titanic/train.csv')    #训练的数据有891个
print(type(train))
test=pd.read_csv('./titanic/test.csv')      #测试数据有418个
PassengerId=test['PassengerId']
all_data=pd.concat([train,test],ignore_index=True)   #合并两数据集，默认纵向合并，默认并集合并，忽略原索引


"""
2.数据分析,使用统计学和绘图。barplot:条状图，kdeplot：内核分布估计图，countplot：计数图
目的：了解数据之间的相关性，为构造特征工程以及模型建立做准备
"""
# print(train.head())   #打印出数据集前五行的数据
# print(train.info())     #打印出数据集的信息，包括每个字段的信息，发现有些字段有空缺
# print(train['Survived'].value_counts())   #打印出训练数据中存活的人数和死亡的人数
# sns.barplot(x='Sex',y='Survived',data=train)    #画出不同性别的生存率图，发现女性幸存率要高于男性
# sns.barplot(x='Pclass',y='Survived',data=train)     #画出不同客舱等级（社会等级）的生存率图，发现客舱等级越高的人生存率也越高。
# sns.barplot(x='SibSp',y='Survived',data=train)      #画出旁系亲友人数（配偶及兄弟姐妹）的生存率图，发现旁系亲友数适中的人生存率更高
# sns.barplot(x='Parch',y='Survived',data=train)      #画出直系亲属人数（父母与子女）的生存率图，发现直系亲属人数适中的人存活率更高
# 绘制条件关系的多网格图,年龄和存活情况的网格图
# facet=sns.FacetGrid(train,hue='Survived',aspect=2)#画出轮廓,
# facet.map(sns.kdeplot,'Age',shade=True)#填充内容，
# facet.set(xlim=(0,train['Age'].max()))#设置x轴的取值范围
# facet.add_legend()#加注释
# plt.xlabel('Age')#横坐标
# plt.ylabel('density')#纵坐标
# plt.show()


# sns.countplot('Embarked',hue="Survived",data=train)#画出Embarked登港港口（S,C,Q）与生存情况的计数图。发现C地的存活率较高
# plt.show()

# 不同称呼的乘客幸存率不同，新增Title特征，从名字中提取乘客的称呼，归纳为六类
all_data['Title']=all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
# print(all_data['Title'].value_counts())
Title_Dict={}
Title_Dict.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'Officer'))#将这些称呼归纳为事业人员
Title_Dict.update(dict.fromkeys(['Don','Sir','the Countess','Dona','Lady'],'Royalt'))#将这些称呼归纳为与皇室有关的人
Title_Dict.update(dict.fromkeys(['Mme','Ms','Mrs'],'Mrs'))#将这些称呼归纳为贵妇
Title_Dict.update(dict.fromkeys(['Mlle','Miss'],'Miss'))#将这些称呼归纳为大户人家小姐
Title_Dict.update(dict.fromkeys(['Mr'],'Mr'))#将这些称呼归纳为绅士
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'],'Master'))#将这些称呼归纳为大师
all_data['Title']=all_data['Title'].map(Title_Dict)#将数据集中的Title映射为归纳的称呼
# print(all_data['Title'].value_counts())
# sns.barplot(x='Title',y='Survived',data=all_data)
# plt.show()

#家庭人数为2到4的乘客生存率较高，计算方式为直系和旁系加自己
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
# sns.barplot(x='FamilySize',y='Survived',data=all_data)
#按生存率把FamilySize分成三类(2,1,0)，构成FamilyLabel特征
def Fam_label(s):
    if (s>=2)&(s<=4):
        return 2
    elif ((s>4)&(s<=7))|(s==1):
        return 1
    elif (s>7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
# sns.barplot(x='FamilyLabel',y='Survived',data=all_data)
# plt.show()


#不同甲板的乘客幸存率不同，新增Deck(甲板号)特征，先把Cabin空缺值填充为Unknown，再提取Cabin（客舱编号）中的首字母构成甲板号
all_data['Cabin']=all_data['Cabin'].fillna('Unknown')#将Cabin中空缺值填充值为Unknown，应该是住在底层的人
all_data['Deck']=all_data['Cabin'].str.get(0)  #C123,C85
# sns.barplot(x='Deck',y='Survived',data=all_data)
# plt.show()


#发现2至4人共票号的乘客幸存率较高，统计每个乘客的共号票
Ticket_Count=dict(all_data['Ticket'].value_counts())
all_data['TicketGroup']=all_data['Ticket'].apply(lambda x:Ticket_Count[x])
# sns.barplot(x='TicketGroup',y="Survived",data=all_data)
# plt.show()
#按生存率把TicketGroup分成三类,2-4为2，4-8，1为1，大于8为0
def Ticket_Label(s):
    if (s>=2)&(s<=4):
        return 2
    elif ((s>4)&(s<=8))|(s==1):
        return 1
    elif (s>8):
        return 0
all_data['TicketGroup']=all_data['TicketGroup'].apply(Ticket_Label)
# sns.barplot(x='TicketGroup',y='Survived',data=all_data)
# plt.show()


"""
3.数据清洗
sklearn库：支持包括分类、回归、降维和聚类四大机器学习算法
，还包括了特征提取、数据处理和模型评估者三大模块。是Scipy的扩展，建立在Numpy和matplolib库的基础上
数据清洗:将数据中缺失的值填充
"""

#Age特征的缺失量比较大，填充数据，用Sex，Title，Pclass三个特征构建随机森林模型
# from sklearn.ensemble import RandomForestRegressor
# age_df=all_data[['Age','Pclass','Sex','Title']]
# age_df=pd.get_dummies(age_df)#对数据进行one-hot编码，DataFrame中数字部分不会进行one-hot编码，但Series会对数字进行one-hot编码
# # known_age=age_df[age_df.Age.notnull()].as_matrix()#报错AttributeError: 'DataFrame' object has no attribute 'as_matrix'
# known_age=age_df[age_df.Age.notnull()].iloc[:,:].values #把dataframe数据格式转变为矩阵形式,age_df[age_df.Age.notnull()].values
# unknown_age=age_df[age_df.Age.isnull()].iloc[:,:].values
# y=known_age[:,0]#取所有行的第0列数据，作标签
# X=known_age[:,1:]#取所有行的第1列数据以后的所有数据，作特征
# rfr=RandomForestRegressor(random_state=0,n_estimators=100,n_jobs=-1)
# rfr.fit(X,y)
# predictedAges=rfr.predict(unknown_age[:,1::])
# print(predictedAges)
# all_data.loc[(all_data.Age.isnull()),'Age']=predictedAges



#对Age的缺失值，采用平均值填充

# all_data['Age']=all_data['Age'].fillna(all_data['Age'].mean())




# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.3)
# rfr=RandomForestClassifier(random_state=0,n_estimators=100,n_jobs=-1)#参数(random:控制所使用样本的自举的随机性，n_estimators=树的数量即评估器的数量，n_job:表示并行的作业数量，-1表示使用所以处理器)
# rfr.fit(Xtrain.astype('int'),Ytrain.astype('int'))  #从训练集中构建一个树的森林
# print(rfr.score(Xtest.astype('int'),Ytest.astype('int')))
# rfr=RandomForestRegressor(random_state=0,n_estimators=100,n_jobs=-1)#0.36863622792203976
# rfr.fit(Xtrain,Ytrain)
# print(rfr.score(Xtest,Ytest))
# rfr=RandomForestRegressor(random_state=0,n_estimators=200,n_jobs=-1)#0.4196437660659139
# rfr.fit(Xtrain,Ytrain)
# print(rfr.score(Xtest,Ytest))
# rfr=RandomForestRegressor(random_state=0,n_estimators=300,n_jobs=-1)#0.3556801034247512
# rfr.fit(Xtrain,Ytrain)
# print(rfr.score(Xtest,Ytest))





#港口号缺失填充.
# Embarked缺失量为2，缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，因为Embarked为C且Pclass为1的乘客的Fare中位数为80，所以缺失值填充为C。
# print(all_data[all_data['Embarked'].isnull()])
# print(all_data.groupby(['Pclass','Embarked']).Fare.median())
all_data['Embarked']=all_data['Embarked'].fillna('C')

#票价填充。
#Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，Pclass为3，所以用Embarked为S，Pclass为3的乘客的Fare中位数填充
fare=all_data[(all_data['Embarked']=='S')&(all_data['Pclass']==3)].Fare.median()   #fare=8.05
all_data['Fare']=all_data['Fare'].fillna(fare)

#利用回归模型预测缺失值
# from sklearn.linear_model import LinearRegression
# data=all_data[['Age','Pclass','Sex','Parch','SibSp','Fare']]
# data['Sex']=[1 if x=='male' else 0 for x in data['Sex']]
# test_data=data[data['Age'].isnull()]
# data.dropna(inplace=True)  #返回删除了缺失值的数据
#
# y_train=data['Age']
# X_train=data.drop('Age',axis=1)
# X_test=test_data.drop('Age',axis=1)
#
# model=LinearRegression()
# model.fit(X_train,y_train)
# y_pred=model.predict(X_test)
# print(y_pred)

import datawig

data=all_data[['PassengerId','Age','Pclass','Sex','Parch','SibSp','Fare']]
data['Sex']=[1 if x=='male' else 0 for x in data['Sex']]
test_data=data[data['Age'].isnull()]
test_data=test_data.drop('Age',axis=1)
train_data=data[data['Age'].notnull()]

df_train,df_test=datawig.utils.random_split(all_data)
#初始化一个简单的imputer模型
imputer=datawig.SimpleImputer(
    input_columns=['Pclass','SibSp','Parch','Fare','Sex'], #输入的列
    output_column='Age', #我们要为其注入值的列
    output_path='imputer_model' #存储模型数据和度量
)
imputer.fit(train_df=train_data,num_epochs=50)#拟合训练数据的模型
imputed=imputer.predict(test_data,imputation_suffix="") #输入丢失的值并返回原始的数据模型和预测

imputed=pd.DataFrame(imputed)

all_data.loc[(all_data.Age.isnull()),'Age']=imputed.Age


# all_data['Age']=all_data['PassengerId'].apply(lambda x:imputed['Age'] if x==imputed["PassengerId"] else 0)
# print(all_data['Age'].isnull())
# all_data.to_csv('./aldata.csv')



#同组识别
#把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性。
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())#将每个人的姓氏提取出来放在新的一列
Surname_Count=dict(all_data['Surname'].value_counts())#key:Surname value:同组人数
all_data['FamilyGroup']=all_data['Surname'].apply(lambda x:Surname_Count[x])#将每个人的同组人数放在新的一列
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2)&((all_data['Age']<=12)|(all_data['Sex']=='female'))]#将妇女儿童组的数据提取出来
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2)&(all_data['Age']>12)&(all_data['Sex']=='male')]#将成年男性组的数据提取出来


Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
# print(Female_Child)
Female_Child.columns=['GroupCount']
# print(Female_Child)
sns.barplot(x=Female_Child.index,y=Female_Child['GroupCount']).set_xlabel("AverageSurvived")
plt.show()
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']

#普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的反常组选出来单独处理。
# 女子儿童组的死亡名单
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()    #排列后取平均值是Series类型
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
#男子组幸存名单
Male_Adult_List=Male_Adult_Group.groupby("Surname")['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

#为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改.
train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]

test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex']='male' #将测试数据中与死亡名单上姓氏相同的人的性别改为male
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age']='60' #将测试数据中与死亡名单上姓氏相同的人的年龄改为60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title']='Mr'#将测试数据中与死亡名单上姓氏相同的人的称呼改为Mr
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex']='female'#将测试数据中与存活名单上姓氏相同的人的性别改为female
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age']=5#将测试数据中与存活名单上姓氏相同的人的年龄改为5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title']='Miss'#将测试数据中与存活名单上姓氏相同的人的称呼改为5
#报出警告SettingWithCopyWarning
#A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead


#特征转换
#提取特征，转换为数值变量，划分训练集和数据集
all_data=pd.concat([train,test])#合并数据
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data=pd.get_dummies(all_data)#进行hot-one编码
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)#axis=1表示列
# X=train.as_matrix()[:,1:]
# y=train.as_matrix()[:,0]
X=train.iloc[:,:].values[:,1:]
y=train.iloc[:,:].values[:,0]



"""
4.建模和优化
用网格搜索自动化选取最优参数，事实上我用网格搜索得到的最优参数是n_estimators = 28，max_depth = 6。
但是参考另一篇Kernel把参数改为n_estimators = 26，max_depth = 6之后交叉验证分数和kaggle评分都有略微提升。
"""

#参数优化
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
#
# pipe=Pipeline([('select',SelectKBest(k=20)),
#                ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])
#
# param_test = {'classify__n_estimators':list(range(20,50,2)),
#               'classify__max_depth':list(range(3,60,3))}
# gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10)#搜索最佳的参数
# gsearch.fit(X,y)
# print(gsearch.best_params_,gsearch.best_score_)



#训练模型
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)   #根据K个最高的分数选择特征
clf = RandomForestClassifier(random_state = 10, warm_start = True,
                                  n_estimators = 26,
                                  max_depth = 9,
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

#交叉验证,用来观察模型的稳定性的方法
#将数据划分为n份，依次使用其中一份作为测试集，其他n-1份作为训练集，多次计算模型的精确性来评估模型的平均准确程度
#因为训练集和测试集的划分会干扰模型的结果，因此用交叉验证n次的结果求出的平均值，是对模型效果的一个更好度量
from sklearn import model_selection,metrics
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)#参数(实例化模型，完整的数据集，完整的数据集标签，数据分成10份)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))


#预测
predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("./submission2.csv", index=False)


# plt.show()



