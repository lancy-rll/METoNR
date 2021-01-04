import numpy as np
import logging
user_num=82698
news_num=2743
nc_num=51
n_topic=50
na_num=473
city_num=2094
os_num=9
device_num=3
class loadDataset(object):
    def __init__(self,path):
        self.train=self.load_train(path+'edge/uns_train')
        self.test = self.load_test(path + 'edge/uns_test')
        self.user_clicked=self.load_user_newslist(path+'edge/uns')
        self.user_clicked_train = self.load_user_newslist_train(path + 'edge/uns_train')
        self.user_clicked_test = self.load_user_newslist_test(path + 'edge/uns_test')
        self.news_neigb_list=self.load_news_neigblist(path+'edge/nn_knn')
        self.news_user_list=self.load_news_userslist_test(path + 'edge/uns_test')
        logging.warning('nun_neigb_list')
        self.nun_neigb_list=self.load_nun_neigb_list(path+'neigb/nun_neigb')
        logging.warning('ncn_neigb_list')
        self.ncn_neigb_list=self.load_ncn_neigb_list(path+'neigb/ncn_neigb')
        logging.warning('ntn_neigb_list')
        self.ntn_neigb_list=self.load_ntn_neigb_list(path+'neigb/ntn_neigb')
        logging.warning('nan_neigb_list')
        self.nan_neigb_list=self.load_nan_neigb_list(path+'neigb/nan_neigb')
        logging.warning('unu_neigb_list')
        self.unu_neigb_list=self.load_unu_neigb_list(path+'neigb/unu_neigb')
        logging.warning('ucu_neigb_list')
        self.ucu_neigb_list=self.load_ucu_neigb_list(path+'neigb/ucu_neigb')
        logging.warning('udu_neigb_list')
        self.udu_neigb_list=self.load_udu_neigb_list(path+'neigb/udu_neigb')
        logging.warning('uou_neigb_list')
        self.uou_neigb_list=self.load_uou_neigb_list(path+'neigb/uou_neigb')


    # def load_train(self,filename):
    #     train = []
    #     with open(filename,'r',encoding='utf-8') as f:
    #         for line in f:
    #             line=line.strip('\n').split()
    #             u,n=int(line[0]),int(line[1])
    #             train.append(([u,n]))
    #     return train
    def load_train(self,filename):
        train = []
        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip('\n').split('|')
                u,n_list=int(line[0]),line[1].split(',')
                for n in n_list[-2:]:
                    train.append([u,int(n)])
        return train
    def load_test(self,filename):
        test = []
        count,count_u=0,0
        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip('\n').split('|')
                u,n_list=int(line[0]),line[1].split(',')
                count_u += 1
                if count_u==10001:
                    print('test u_10000',u)
                    print('test count', count)
                    break
                for n in n_list:
                    test.append([u,int(n)])
                    count+=1
        return test
    def load_user_newslist(self,filename):
        u_nlist=[]
        u_nlist.append([0])
        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                a=line.split('|')[1].split(',')
                t_list=[]
                for i in a:
                    t_list.append(int(i))
                u_nlist.append(t_list)
        return u_nlist

    def load_user_newslist_train(self,filename):
        u_nlist=[]
        u_nlist.append([0])
        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                a=line.split('|')[1].split(',')
                t_list=[]
                for i in a:
                    t_list.append(int(i))
                u_nlist.append(t_list)
        return u_nlist
    def load_user_newslist_test(self,filename):
        u_nlist=[[0]]*82699
        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip('\n').split('|')
                u,a=int(line[0]),line[1].split(',')
                t_list=[]
                for i in a:
                    t_list.append(int(i))
                u_nlist[u]=t_list
        for i in range(len(u_nlist)):
            if u_nlist[i][0]==0:
                u_nlist[i]=u_nlist[i][1:]
        return u_nlist
    def load_news_neigblist(self,filename):
        n_neigb=[[i]*1 for i in range(2744)]

        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip('\n').split()
                n_neigb[int(line[0])].append(int(line[1]))
        return n_neigb

    def load_news_userslist_test(self,filename):
        n_us = [[0] for _ in range(2744)]
        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip('\n').split('|')
                u,n_list=int(line[0]),line[1].split(',')
                for n in n_list:
                    n_us[int(n)].append(u)
        return [u_list[1:] for u_list in n_us]

    def load_nun_neigb_list(self,filename):
        n_neigb=[0]*(news_num+1)
        with open(filename,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip('\n').split('|')
                n_neigb[int(line[0])]=list(map(int,line[1].split()))
        return n_neigb
    def load_ncn_neigb_list(self,filename):
        n_neigb = [0] * (news_num + 1)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('|')
                n_neigb[int(line[0])] = list(map(int, line[1].split()))
        return n_neigb
    def load_ntn_neigb_list(self,filename):
        n_neigb = [0] * (news_num + 1)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('|')
                n_neigb[int(line[0])] = list(map(int, line[1].split()))
        return n_neigb
    def load_nan_neigb_list(self,filename):
        n_neigb = [0] * (news_num + 1)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('|')
                n_neigb[int(line[0])] = list(map(int, line[1].split()))
        return n_neigb
    def load_unu_neigb_list(self,filename):
        u_neigb = [0] * (user_num + 1)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('|')
                u_neigb[int(line[0])] = list(map(int, line[1].split()))
        # user_clicked=self.user_clicked
        # u_neigb = [[i] * 1 for i in range(user_num+1)]
        # for i in range(1,len(u_neigb)):
        #     uid_set=set()
        #     uid_set.add(i)
        #     tmp=[0]*(user_num+1)
        #     for j in range(1,user_num+1):
        #         if i!=j and len(set(user_clicked[i])&set(user_clicked[j]))>0:
        #             tmp[j]=len(set(user_clicked[i])&set(user_clicked[j]))
        #     tmp = sorted(enumerate(tmp), key=lambda x: x[1], reverse=True)
        #     tmp = [i[0] for i in tmp]
        #
        #     if len(tmp)>=50:
        #         u_neigb[i]+=tmp[:50]
        #     else:
        #         u_neigb[i] += tmp
        return u_neigb
    def load_ucu_neigb_list(self,filename):
        u_neigb = [0] * (user_num + 1)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('|')
                u_neigb[int(line[0])] = list(map(int, line[1].split()))
        # cu = [[0]] * (city_num + 1)
        # with open(filename, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip('\n').split()
        #         cu[int(line[1])].append(int(line[0]))
        # cu = [i[1:] for i in cu]
        #
        # u_neigb = [[i] * 1 for i in range(user_num + 1)]
        # for i in range(1, len(u_neigb)):
        #     uid_set = set()
        #     uid_set.add(i)
        #     tmp = []
        #     for j in range(1, city_num + 1):
        #         if i in cu[j]:
        #             t = list(uid_set ^ set(cu[j]))
        #             tmp += t
        #     if len(set(tmp)) >= 50:
        #         u_neigb[i] += list(set(tmp))[:50]
        #     else:
        #         u_neigb[i] += list(set(tmp))

        return u_neigb
    def load_udu_neigb_list(self,filename):
        u_neigb = [0] * (user_num + 1)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('|')
                u_neigb[int(line[0])] = list(map(int, line[1].split()))
        # du = [[0]] * (device_num + 1)
        # with open(filename, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip('\n').split()
        #         du[int(line[1])].append(int(line[0]))
        # du = [i[1:] for i in du]
        #
        # u_neigb = [[i] * 1 for i in range(user_num + 1)]
        # for i in range(1, len(u_neigb)):
        #     uid_set = set()
        #     uid_set.add(i)
        #     tmp = []
        #     for j in range(1, device_num + 1):
        #         if i in du[j]:
        #             t = list(uid_set ^ set(du[j]))
        #             tmp += t
        #     if len(set(tmp)) >= 50:
        #         u_neigb[i] += list(set(tmp))[:50]
        #     else:
        #         u_neigb[i] += list(set(tmp))

        return u_neigb
    def load_uou_neigb_list(self,filename):
        u_neigb = [0] * (user_num + 1)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('|')
                u_neigb[int(line[0])] = list(map(int, line[1].split()))
        # ou = [[0]] * (os_num + 1)
        # with open(filename, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip('\n').split()
        #         ou[int(line[1])].append(int(line[0]))
        # ou = [i[1:] for i in ou]
        #
        # u_neigb = [[i] * 1 for i in range(user_num + 1)]
        # for i in range(1, len(u_neigb)):
        #     uid_set = set()
        #     uid_set.add(i)
        #     tmp = []
        #     for j in range(1, os_num + 1):
        #         if i in ou[j]:
        #             t = list(uid_set ^ set(ou[j]))
        #             tmp += t
        #     if len(set(tmp)) >= 50:
        #         u_neigb[i] += list(set(tmp))[:50]
        #     else:
        #         u_neigb[i] += list(set(tmp))

        return u_neigb






