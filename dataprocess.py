import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import defaultdict,OrderedDict
import numpy as np
#inputfilename
prefix='F:/newsdataset/data/'
prefixe='F:/newsdataset/data/edge/'
prefix_neigb='F:/newsdataset/data/neigb/'
u_info=prefix+'user_info'
u_news=prefix+'un'
u_city=prefix+'city'
u_country=prefix+'country'
u_os=prefix+'os'
u_dt=prefix+'deviceType'
n_info=prefix+'news_info'
n_author=prefix+'author'
n_categoryc=prefix+'categoryc'
n_categoryf=prefix+'categoryf'
uu_knn=prefixe+'uns'
nn_knn=prefixe+'nus'
#outputfilename
uct_out=prefixe+'uct'
uc_out=prefixe+'uc'
uo_out=prefixe+'uo'
udt_out=prefixe+'udt'
na_out=prefixe+'na'
nca_out=prefixe+'nca'
ccf_out=prefixe+'ccf'
uu_knn_out=prefixe+'uu_knn'
nn_knn_out=prefixe+'nn_knn'


#user-city
uct_dic=defaultdict(int)
#user-country
uc_dic=defaultdict(int)
#user-os
uo_dic=defaultdict(int)
#user-deviceType
udt_dic=defaultdict(int)
#news-author['']
na_dic=defaultdict(list)
#news-category
nca_dic=defaultdict(int)
#categoryc-categoryf
ccf_dic=defaultdict(int)

user_num=82698
news_num=2743
nc_num=51
n_topic=50
na_num=473
city_num=2094
os_num=9
device_num=3


categoryc2idx=defaultdict(int)
categoryf2idx=defaultdict(int)
#user
def user_city():
    city2idx=defaultdict(int)
    idx2city=defaultdict(str)
    idx=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            ct=line.strip('\n').split('|')[1]
            if ct not in city2idx and ct!='other':
                city2idx[ct],idx2city[idx]=idx,ct
                idx+=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            user_id,city=line.strip('\n').split('|')[0],line.strip('\n').split('|')[1]
            if city!='other':
                city_idx=city2idx[city]
                uct_dic[user_id]=city_idx
    with open(uct_out,'w',encoding='utf-8') as f:
        for user,city in uct_dic.items():
            f.write(str(user)+'\t'+str(city)+'\n')
def user_country():
    country2idx=defaultdict(int)
    idx2country=defaultdict(str)
    idx=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            c=line.strip('\n').split('|')[2]
            if c not in country2idx and c!='other':
                country2idx[c],idx2country[idx]=idx,c
                idx+=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            user_id,country=line.strip('\n').split('|')[0],line.strip('\n').split('|')[2]
            if country!='other':
                country_idx=country2idx[country]
                uc_dic[user_id]=country_idx
    with open(uc_out,'w',encoding='utf-8') as f:
        for user,country in uc_dic.items():
            f.write(str(user)+'\t'+str(country)+'\n')
def user_os():
    os2idx=defaultdict(int)
    idx2os=defaultdict(str)
    idx=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            os=line.strip('\n').split('|')[3]
            if os not in os2idx and os!='other':
                os2idx[os],idx2os[idx]=idx,os
                idx+=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            user_id,os=line.strip('\n').split('|')[0],line.strip('\n').split('|')[3]
            if os!='other':
                os_idx=os2idx[os]
                uo_dic[user_id]=os_idx
    with open(uo_out,'w',encoding='utf-8') as f:
        for user,os in uo_dic.items():
            f.write(str(user)+'\t'+str(os)+'\n')
def user_deviceType():
    dt2idx=defaultdict(int)
    idx2dt=defaultdict(str)
    idx=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            dt=line.strip('\n').split('|')[4]
            if dt not in dt2idx and dt!='other':
                dt2idx[dt],idx2dt[idx]=idx,dt
                idx+=1
    with open(u_info,'r',encoding='utf-8') as f:
        for line in f:
            user_id,dt=line.strip('\n').split('|')[0],line.strip('\n').split('|')[4]
            if dt!='other':
                dt_idx=dt2idx[dt]
                udt_dic[user_id]=dt_idx
    with open(udt_out,'w',encoding='utf-8') as f:
        for user,deviceType in udt_dic.items():
            f.write(str(user)+'\t'+str(deviceType)+'\n')
#news
def news_author():
    author2idx=defaultdict(int)
    idx=1
    with open(n_author,'r',encoding='utf-8') as f:
        for line in f:
            a=line.strip('\n')
            author2idx[a]=idx
            idx+=1
    with open(n_info,'r',encoding='utf-8') as f:
        for line in f:
            news_id,a_str=line.strip('\n').split('|')[0],line.strip('\n').split('|')[3]
            a_list=a_str.split(',')
            na_dic[news_id]=a_list
    with open(na_out, 'w', encoding='utf-8') as f:
        for news,at_list in na_dic.items():
            for a in at_list:
                if a in author2idx:
                    a_idx = author2idx[a]
                    f.write(str(news) + '\t' + str(a_idx) + '\n')

def news_subcategory():
    idx=1
    with open(n_categoryc,'r',encoding='utf-8') as f:
        for line in f:
            a=line.strip('\n')
            categoryc2idx[a]=idx
            idx+=1
    with open(n_info,'r',encoding='utf-8') as f:
        for line in f:
            news_id,c_str=line.strip('\n').split('|')[0],line.strip('\n').split('|')[1]
            if c_str!='other':
                a_list=c_str.split(',')
                nca_dic[news_id]=a_list[1]
    with open(nca_out, 'w', encoding='utf-8') as f:
        for news,category in nca_dic.items():
            ca_idx=categoryc2idx[category]
            f.write(str(news)+'\t'+str(ca_idx)+'\n')
def subcategory_category():
    idx=1
    with open(n_categoryf,'r',encoding='utf-8') as f:
        for line in f:
            a=line.strip('\n')
            categoryf2idx[a]=idx
            idx+=1
    with open(n_info,'r',encoding='utf-8') as f:
        for line in f:
            c_str=line.strip('\n').split('|')[1]
            if c_str!='other':
                a_list=c_str.split(',')
                cf_idx,cc_idx=categoryf2idx[a_list[0]],categoryc2idx[a_list[1]]
                ccf_dic[cc_idx]=cf_idx
    with open(ccf_out, 'w', encoding='utf-8') as f:
        for cc,cf in ccf_dic.items():
            f.write(str(cc)+'\t'+str(cf)+'\n')
def knn():
    un_dic=defaultdict(list)
    nu_dic=defaultdict(list)

    uu_dic=defaultdict(list)
    nn_dic=defaultdict(list)

    uu_score_dic=defaultdict(dict)
    nn_score_dic=defaultdict(dict)

    count=1

    #k,v all int
    with open(u_news,'r',encoding='utf-8') as f:
        for line in f:
            user_id,news_id=line.strip('\n').split()[0:2]
            user_id,news_id=int(user_id),int(news_id)
            un_dic[user_id].append(news_id)
            nu_dic[news_id].append(user_id)
            # count+=1
            # if count>100:
            #     break
    f.close()
    for user_id,news_list in un_dic.items():
        for news in news_list:
            uu_dic[user_id]+=nu_dic[news]
        uu_dic[user_id]=list(set(uu_dic[user_id]))
    for news_id,user_list in nu_dic.items():
        for user in user_list:
            nn_dic[news_id]+=un_dic[user]
        nn_dic[news_id]=list(set(nn_dic[news_id]))
    for user_id,user_list in uu_dic.items():
        u_nlist=un_dic[user_id]
        u_n_num=len(u_nlist)
        for user2_id in user_list:
            u2_nlist=un_dic[user2_id]
            u2_n_num=len(u2_nlist)

            co_num=len([ns for ns in u_nlist if ns in u2_nlist])
            uu_score_dic[user_id][(user_id,user2_id)]=co_num/np.sqrt(u_n_num*u2_n_num)
    for news_id,news_list in nn_dic.items():
        n_ulist=nu_dic[news_id]
        n_u_num=len(n_ulist)
        for news2_id in news_list:
            n2_ulist=nu_dic[news2_id]
            n2_u_num=len(n2_ulist)

            co_num=len([us for us in n_ulist if us in n2_ulist])
            nn_score_dic[news_id][(news_id,news2_id)]=co_num/np.sqrt(n_u_num*n2_u_num)
    with open(uu_knn_out,'w',encoding='utf-8') as f:
        uu_score_dic=OrderedDict(sorted(uu_score_dic.items(),key=lambda x:x[0]))
        for user_id,_ in uu_score_dic.items():
            sorted_uu=sorted(uu_score_dic[user_id].items(),key=lambda x:x[1],reverse=True)[1:]
            for uu2val in sorted_uu:
                u1,u2,val=uu2val[0][0],uu2val[0][1],uu2val[1]
                if float(val)>0.4:
                    f.write(str(u1)+'\t'+str(u2)+'\t'+str(val)+'\n')
                else:
                    break
    f.close()
    with open(nn_knn_out,'w',encoding='utf-8') as f:
        nn_score_dic=OrderedDict(sorted(nn_score_dic.items(),key=lambda x:x[0]))
        for news_id,_ in nn_score_dic.items():
            sorted_nn=sorted(nn_score_dic[news_id].items(),key=lambda x:x[1],reverse=True)[1:]
            for nn2val in sorted_nn:
                n1,n2,val=nn2val[0][0],nn2val[0][1],nn2val[1]
                if float(val)>0.4:
                    f.write(str(n1)+'\t'+str(n2)+'\t'+str(val)+'\n')
                else:
                    break
    f.close()
def user_knn():
    uid_n=[0]*82699
    u_nlist=[0]*82699

    with open(uu_knn,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip('\n').split('|')
            n_list=list(map(int,line[1].split(',')))
            uid_n[int(line[0])]=len(n_list)
            u_nlist[int(line[0])]=set(n_list)


    for user_id,news_num in enumerate(uid_n):
        with open(uu_knn, 'r', encoding='utf-8') as f:
            if user_id!=0:
                uu_score = [0]*82699
                for line in f:
                    line=line.strip('\n').split('|')
                    user2_id=int(line[0])
                    if user_id!=user2_id:
                        n2_list=list(map(int,line[1].split(',')))
                        n_set=u_nlist[user_id]
                        co_num=len(n_set & set(n2_list))
                        uu_score[user2_id]=co_num / np.sqrt(uid_n[user_id] * uid_n[user2_id])
                uu_score=sorted(enumerate(uu_score), key=lambda x: x[1], reverse=True)
                with open(uu_knn_out,'a',encoding='utf-8') as nf:
                    for uid,score in uu_score:
                        # nf.write(str(user_id) + '\t' + str(uid) + '\t' + str(score) + '\n')
                        if score>0.53:
                            nf.write(str(user_id)+'\t'+str(uid)+'\t'+str(score)+'\n')
                        else:
                            break
def news_knn():
    nid_n=[0]*2744
    n_ulist=[0]*2744
    with open(nn_knn,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip('\n').split('|')
            u_list=list(map(int,line[1].split(',')))
            nid_n[int(line[0])]=len(u_list)
            n_ulist[int(line[0])]=set(u_list)

    for news_id,user_num in enumerate(nid_n):
        with open(nn_knn, 'r', encoding='utf-8') as f:
            if news_id!=0:
                nn_score = [0]*2744
                for line in f:
                    line=line.strip('\n').split('|')
                    news2_id=int(line[0])
                    if news_id!=news2_id:
                        u2_list=list(map(int,line[1].split(',')))
                        u_set=n_ulist[news_id]
                        co_num=len(u_set & set(u2_list))
                        nn_score[news2_id]=co_num / np.sqrt(nid_n[news_id] * nid_n[news2_id])
                nn_score=sorted(enumerate(nn_score), key=lambda x: x[1], reverse=True)
                with open(nn_knn_out,'a',encoding='utf-8') as nf:
                    for nid,score in nn_score:
                        # nf.write(str(news_id) + '\t' + str(nid) + '\t' + str(score) + '\n')
                        if score>0.25:
                            nf.write(str(news_id)+'\t'+str(nid)+'\t'+str(score)+'\n')
                        else:
                            break

def user_newslist():
    u_nlist=[]
    u_nlist.append([0])
    with open(prefix+'edge/uns','r',encoding='utf-8') as f:
        for line in f:
            a=line.split('|')[1].split(',')
            t_list=[]
            for i in a:
                t_list.append(int(i))
            u_nlist.append(t_list)
    return u_nlist
def nun_neigb_list(u_nlist):
    user_clicked=u_nlist
    n_neigb = [[i] * 1 for i in range(2744)]
    for i in range(1,len(n_neigb)):
        nid_set=set()
        nid_set.add(i)
        tmp=[]
        for j in range(1,user_num+1):
            if i in user_clicked[j]:
                t=list(nid_set^set(user_clicked[j]))
                tmp+=t
        if len(set(tmp))>=50:
            n_neigb[i]+=list(set(tmp))[:50]
        else:
            n_neigb[i] += list(set(tmp))
    with open(prefix_neigb+'nun_neigb','w',encoding='utf-8') as f:
        for i in range(len(n_neigb)):
            c=map(str,n_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')

def ncn_neigb_list():
    cn=[[0]]*(nc_num+1)
    with open(prefixe+'nca', 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip('\n').split()
            cn[int(line[1])].append(int(line[0]))
    cn=[i[1:] for i in cn]

    n_neigb = [[i] * 1 for i in range(2744)]
    for i in range(1,len(n_neigb)):
        nid_set=set()
        nid_set.add(i)
        tmp=[]
        for j in range(1,nc_num+1):
            if i in cn[j]:
                t=list(nid_set^set(cn[j]))
                tmp+=t
        if len(set(tmp))>=50:
            n_neigb[i]+=list(set(tmp))[:50]
        else:
            n_neigb[i] += list(set(tmp))
    with open(prefix_neigb+'ncn_neigb','w',encoding='utf-8') as f:
        for i in range(len(n_neigb)):
            c=map(str,n_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')

def ntn_neigb_list():
    tn=[[0]]*(n_topic+1)
    with open(prefixe+'nt', 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip('\n').split()
            tn[int(line[1])].append(int(line[0]))
    tn=[i[1:] for i in tn]

    n_neigb = [[i] * 1 for i in range(2744)]
    for i in range(1,len(n_neigb)):
        nid_set=set()
        nid_set.add(i)
        tmp=[]
        for j in range(1,n_topic+1):
            if i in tn[j]:
                t=list(nid_set^set(tn[j]))
                tmp+=t
        if len(set(tmp))>=50:
            n_neigb[i]+=list(set(tmp))[:50]
        else:
            n_neigb[i] += list(set(tmp))
    with open(prefix_neigb+'ntn_neigb','w',encoding='utf-8') as f:
        for i in range(len(n_neigb)):
            c=map(str,n_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')

def nan_neigb_list():
    an=[[0]]*(na_num+1)
    with open(prefixe+'na', 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip('\n').split()
            an[int(line[1])].append(int(line[0]))
    an=[i[1:] for i in an]

    n_neigb = [[i] * 1 for i in range(2744)]
    for i in range(1,len(n_neigb)):
        nid_set=set()
        nid_set.add(i)
        tmp=[]
        for j in range(1,na_num+1):
            if i in an[j]:
                t=list(nid_set^set(an[j]))
                tmp+=t
        if len(set(tmp))>=50:
            n_neigb[i]+=list(set(tmp))[:50]
        else:
            n_neigb[i] += list(set(tmp))
    with open(prefix_neigb+'nan_neigb','w',encoding='utf-8') as f:
        for i in range(len(n_neigb)):
            c=map(str,n_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')
def unu_neigb_list(u_nlist):
    user_clicked=u_nlist
    u_neigb = [[i] * 1 for i in range(user_num+1)]
    for i in range(1,len(u_neigb)):
        uid_set=set()
        uid_set.add(i)
        tmp=[0]*(user_num+1)
        for j in range(1,user_num+1):
            if i!=j and len(set(user_clicked[i])&set(user_clicked[j]))>0:
                tmp[j]=len(set(user_clicked[i])&set(user_clicked[j]))
        tmp = sorted(enumerate(tmp), key=lambda x: x[1], reverse=True)
        tmp = [i[0] for i in tmp if i[1]>0]

        if len(tmp)>=50:
            u_neigb[i]+=tmp[:50]
        else:
            u_neigb[i] += tmp
    with open(prefix_neigb+'unu_neigb','w',encoding='utf-8') as f:
        for i in range(len(u_neigb)):
            c=map(str,u_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')

def ucu_neigb_list():
    cu = [[0]*1 for _ in range(city_num + 1)]
    with open(prefixe+'uct', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split()
            cu[int(line[1])].append(int(line[0]))
    cu = [i[1:] for i in cu]

    u_neigb = [[i] * 1 for i in range(user_num + 1)]
    for i in range(1, len(u_neigb)):
        uid_set = set()
        uid_set.add(i)
        tmp = []
        for j in range(1, city_num + 1):
            if i in cu[j]:
                t = list(uid_set ^ set(cu[j]))
                tmp += t
        if len(set(tmp)) >= 50:
            u_neigb[i] += list(set(tmp))[:50]
        else:
            u_neigb[i] += list(set(tmp))
    with open(prefix_neigb+'ucu_neigb','w',encoding='utf-8') as f:
        for i in range(len(u_neigb)):
            c=map(str,u_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')

def udu_neigb_list():
    du = [[0]*1 for _ in range(device_num + 1)]
    with open(prefixe+'udt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split()
            du[int(line[1])].append(int(line[0]))
    du = [i[1:] for i in du]

    u_neigb = [[i] * 1 for i in range(user_num + 1)]
    for i in range(1, len(u_neigb)):
        uid_set = set()
        uid_set.add(i)
        tmp = []
        for j in range(1, device_num + 1):
            if i in du[j]:
                t = list(uid_set ^ set(du[j]))
                tmp += t
        if len(set(tmp)) >= 50:
            u_neigb[i] += list(set(tmp))[:50]
        else:
            u_neigb[i] += list(set(tmp))
    with open(prefix_neigb+'udu_neigb','w',encoding='utf-8') as f:
        for i in range(len(u_neigb)):
            c=map(str,u_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')

def uou_neigb_list():
    ou = [[0]*1 for _ in range(os_num + 1)]
    with open(prefixe+'uo', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split()
            ou[int(line[1])].append(int(line[0]))
    ou = [i[1:] for i in ou]

    u_neigb = [[i] * 1 for i in range(user_num + 1)]
    for i in range(1, len(u_neigb)):
        uid_set = set()
        uid_set.add(i)
        tmp = []
        for j in range(1, os_num + 1):
            if i in ou[j]:
                t = list(uid_set ^ set(ou[j]))
                tmp += t
        if len(set(tmp)) >= 50:
            u_neigb[i] += list(set(tmp))[:50]
        else:
            u_neigb[i] += list(set(tmp))

    with open(prefix_neigb+'uou_neigb','w',encoding='utf-8') as f:
        for i in range(len(u_neigb)):
            c=map(str,u_neigb[i])
            f.write(str(i)+'|'+' '.join(c)+'\n')






if __name__=='__main__':
    # user_city()
    # user_country()
    # user_os()
    # user_deviceType()
    # news_author()
    # news_subcategory()
    # subcategory_category()
    # knn()
    # user_knn()
    # news_knn()
    # user_clicked=user_newslist()
    # nun_neigb_list(user_clicked)
    # ncn_neigb_list()
    # ntn_neigb_list()
    # nan_neigb_list()
    # unu_neigb_list(user_clicked)
    # ucu_neigb_list()
    # udu_neigb_list()
    uou_neigb_list()








