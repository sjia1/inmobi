def compute_user_stat(iem, theta, display_flag):
    categ = ['#business','#sports','#fun_facts','#international','#animals',  '#national', 
            '#health_n_fitness', '#entertainment', '#fashion', '#automotive',   '#food', '#technology', '#travel', 
            '#games', '#music', '#n/a']
    temp = {'categ':categ,'app': np.zeros(len(categ)),'binge_time':np.zeros(len(categ)),
            'avg_binge_time':np.zeros(len(categ)),'bin_rew':np.zeros(len(categ)),
            'avg_bin_rew':np.zeros(len(categ))}
    user_stat = pd.DataFrame(temp)

    for row in iem.iterrows():
        card_temp = row[1]['glance_id'] # get card id 
        temp = card[card.glance_id == card_temp] # search for this card
        if temp.shape[0]!= 1:
            print("something goes wrong")
        else:
            list_temp = temp.category_ids.iloc[0]
            if '#daily_digest' in list_temp:
                list_temp.remove('#daily_digest')
            if list_temp == []:
                user_stat.at[len(categ)-1, 'app'] += 1 # there is a categ called 'n/a'
                user_stat.at[len(categ)-1, 'binge_time'] += row[1]['duration']
            for key in list_temp:
                if key in categ:
                    temp = user_stat[user_stat.categ == key]
                    idx = temp.index[0]
                    user_stat.at[idx, 'app'] += 1/len(list_temp)# num cards of this category
                    user_stat.at[idx, 'binge_time'] += row[1]['duration']/len(list_temp)# binge time shared equally by each categ
                    if row[1]['duration'] > theta:
                        user_stat.at[idx, 'bin_rew'] += 1/len(list_temp) # binary reward                    
                else:  # if this categ is a rare one
                    user_stat.at[len(categ)-1, 'app']+= 1/len(list_temp)
                    user_stat.at[len(categ)-1, 'binge_time'] += row[1]['duration']/len(list_temp) 
                    if row[1]['duration'] > theta:
                        user_stat.at[len(categ)-1, 'bin_rew'] += 1/len(list_temp) # binary reward

    for row in user_stat.iterrows(): # find average binge and binary rewards
        if row[1]['app']>0: 
            user_stat.at[row[0], 'avg_binge_time'] = round(row[1]['binge_time']/(row[1]['app']*10**3),3) # in seconds
            user_stat.at[row[0], 'avg_bin_rew'] = round(row[1]['bin_rew']/(row[1]['app']), 3)
    if display_flag == 'display':
        display(user_stat)
    return user_stat


def compute_user_stat_all(binge_edge, where_to_save):
    # invoked other functions: merge_edges
    user_list = list(binge_edge.user_id_hashed.unique()) # list of distinct users. Assign space for weighted_app. Should be 9376
    num_user = len(user_list)
    temp2 = [[0,0]]*num_user
    temp1 = {'user_id_hashed': user_list,      
           '#business': temp2, '#sports':temp2, '#fun_facts': temp2,  '#international': temp2,  '#animals':temp2,   '#national': temp2,   '#health_n_fitness':temp2,  '#entertainment':temp2,  '#fashion':temp2,  '#automotive':temp2,  '#food':temp2,  '#technology':temp2, '#travel':temp2,  '#games':temp2, '#music': temp2, 'n/a':temp2} # each cell in this df is a LIST = [weighted app, cumulative weighted binge time] 
    user_stat = pd.DataFrame(temp1) # compute the weighted appearance of each category

    ## loop over all rows in binge_edge
    start = time.time()
    for j in range(num_user): # for each of the 9376 users
        user_temp = user_list[j] # get user_id_hashed
        iem = merge_edges(binge_edge, user_temp, 'user','') # merge the multi-edges for this user. iem stands for incident edges merged
        for row in iem.iterrows(): # loop over the merged edges inci to this user
            card_temp = row[1]['glance_id'] # get glance_id
            temp = card[card.glance_id == card_temp] # search for this card in card meta data
            if temp.shape[0] == 1:# if there is a unique card record
                list_temp = temp.category_ids.iloc[0] # a list of categ
                if '#daily_digest' in list_temp: # remove daily digest
                    list_temp.remove('#daily_digest') 
                for key in list_temp:
                    if key in temp1: # there are some very rare categ, just ignore them
                        temp = user_stat.iloc[j][key].copy() # note: cant just write temp = XXX, since RHS is a list 
                        temp[0] += 1/len(list_temp) # thus we need to manually change the cells in this clumsy way
                        temp[1] += row[1]['duration']/len(list_temp) # cum. binge time
                        user_stat.at[j,key] = temp # change the entry in df
                if list_temp == []: # if no categ
                    temp = user_stat.iloc[j]['n/a'].copy() # repeat above
                    temp[0] += 1
                    temp[1] += row[1]['duration']
                    user_stat.at[j,'n/a'] = temp # change the cell 'n/a'

        if j % 10 == 1:
            print("time elapsed for the first ", j-1, "users:", round(time.time()-start, 2), "sec")
        if j % 2000 == 1:
            user_stat.to_csv(where_to_save, sep=',') # partial progress

    # Notification when job complete
    dt_object = datetime.datetime.fromtimestamp(round( time.time()) )
    with open("_NEW TASK COMPLETED AT " + str(dt_object) + " CALLED " + where_to_save, "w") as my_empty_csv:
        pass  # or write something to it already
    user_stat.to_csv(where_to_save, sep=',') # save upon completion
    return user_stat


def embed_card(U, b, v0, eta, max_iter, display_flag, model_name, w):
    # b = reward binary vec
    # U = user vec repn
    # v = card vec
    # output: the most likely embedding of this card
    # alg: projected gradient descent
    v = np.zeros(v0.shape)
    v_best = np.zeros(v0.shape)
    np.copyto(v, v0) # set initial position
    obj_prev = 10**10 # previous objective value 
    obj_best = -10**10 # best obj so far. Set the initial obj to be small since we are maximizing
    pert = 10**(-9) # perturb in case denominator = 0
    stop_rule = 10**(-6) # terminate if change is small
    mag = 1 # if we need to escape local opt, then generate the magnitude of step size randomly
    
    for k in range(max_iter):
        obj = log_likelihood(U,b,v,model_name, w) # compute the current likelihood      
        if obj == float('-inf'): # if obj = -infty, break
            print("obj = -infty! at iter", k)
            break
        if obj_best < obj: # if obj breaks the record
            obj_best = obj
            np.copyto(v_best,v)
            
        grad = 0 # reset gradient
        #mag = 10**np.random.randint(3) 
        ip = np.matmul(U, v) # compute inner products

        ### CASES START ###
        if model_name == 'model1': # Ber(<u,v>)
            for i in range(U.shape[0]): # gradient is a sum over all users 
                coeff = b[i]/(ip[i]+pert) - (1-b[i])/(1-ip[i]+ pert) 
                grad += w[i]* coeff * U[i,:] # recall: w is the user weight
            v += eta * grad * mag # gradient descent
            ## projection: two steps: first project to qst orthant, then to the unit ball
            for i in range(v0.shape[0]): # projection to 1st orthant
                if v[i] < 0:
                    v[i] = 0
            if np.linalg.norm(v) > 1: # if v is not in the unit ball
                v = v/np.linalg.norm(v) # then project to the unit ball
                
        if model_name == 'model2': # Ber(0.5 + 0.5 <u,v>)
            for i in range(U.shape[0]): # gradient is a sum over all users 
                coeff = b[i]/(1+ip[i]+pert) - (1-b[i])/(1-ip[i]+ pert) 
                grad += w[i]*coeff * U[i,:]
            v += eta * grad * mag # gradient descent with noise, in order to escape local opt
            if np.linalg.norm(v) > 1: # if v is not in the unit ball
                v = v/np.linalg.norm(v) # then project to the unit ball
                
        if model_name == 'L_inf': # Ber(<u,v>), L_inf norm
            for i in range(U.shape[0]): # gradient is a sum over all users 
                coeff = b[i]/(ip[i]+pert) - (1-b[i])/(1-ip[i]+ pert) 
                grad += w[i]*coeff * U[i,:]
            v += eta * grad * mag # gradient descent
            for i in range(v0.shape[0]): # projection v to [0,1]^d if outside
                if v[i] < 0:
                    v[i] = 0
                if v[i] > 1:
                    v[i] = 1
        ### CASES END ###
        
        change = obj - obj_prev        
        if abs(change) <= stop_rule: # terminate if this change is tiny
            print("stopping criterion met: change = ", change, "less than", stop_rule)
            break
        obj_prev = obj # the yesterday of tommorrow = today 
        
    if display_flag == 'display':
        if k % 5 == 0:
            print("iteration", k, ", v=", v, ", Current log likelihood", obj)
        print('objective =', np.round(obj_best,3))
        print('v =', np.round(v_best,3))
        print("number iterations", k)
    return [v_best, obj_best]


def find_binge(action, card, where_to_save):
# input: action file (typically a parquet), card
# output: for each card, find the number of binges ('degree') in pq0
    num_card = card.shape[0]
    num_binge = np.zeros(num_card) 
    count = 0
    start = time.time()
    for row in card.iterrows():
#         if count >= 100: # truncate beyond this point
#             break
        if count % 10**2 == 1:
            print("number cards processed", count-1)
            end = time.time()
            print( round(end - start, 2) ) 
            start = time.time() # restart time
        count += 1
        card_name = row[1]['glance_id']
        temp = action[action.glance_id == card_name].session_mode.value_counts() # only BINGE matters, DEFAULT does not.
        if 'BINGE' in temp.index:
            num_binge[count] = temp['BINGE']
    np.savetxt(where_to_save, num_binge, delimiter=",")
    

def find_U(binge_time_cap, theta,display_flag):
    idx = 0 # need to know where to write in U

    for row in iem.iterrows(): # loop over all merged edges incident to this card
        ## compute user vector U
        user_temp = row[1]['user_id_hashed'] # get user id
        temp = user_stat[user_stat.user_id_hashed == user_temp] #search for this user in user_stat. Should be just one row
        if temp.shape[0] != 1: # if search result is not unique
            print("either not unique or not exist")
        else: #### HEART ####
            j = 0 # column index
            [num_merged_edge, cum_binge_time] = [0,0] # want to measure user intensity, this can be done by using user_stat
            for key in temp: # loop over all columns/categories
                if j >= 1 and j <= num_col: # j=0 is the user_id column. skip this column
                    s = temp[key].iloc[0] # the list stored is a str of the form '[a,b,c]'
                    list_temp = remove_quote(s) # so we need to remove the quote. Now list_temp is a list floats 
                    if list_temp[0] >= 5: # Embedding rule: if num app >=5, then write the avg binge time into U
                        U[idx, j-1] = list_temp[1]/list_temp[0] # avg binge time (per app)
                    num_merged_edge += list_temp[0] # weighted app of this key
                    cum_binge_time += list_temp[1] # (merged) binge_time of this key
                j += 1

        if num_merged_edge != 0:
            user_intensity[idx] = cum_binge_time/num_merged_edge # binge time per card
        true_binge_time[idx] = row[1]['duration'] # true binge time
        if row[1]['duration'] > theta: # compute reward vector b
            rew_vec[idx] = 1
        if display_flag == 'display': # display. Can be turned off
            print("user id:", user_temp)
            print("cum. binge time:", round(cum_binge_time/10**3,3), "sec.") 
            print("num_merged_edge:", num_merged_edge)
            print("avg binge time per (merged) edge ('user_intensity'):", round(user_intensity[idx]/10**3,3), "sec.")
            print("details:")
            display(temp.iloc[0])
            print("*"*50)
        idx += 1

    for i in range(num_lnb):# remember to normalize each row of U (wrt L1 norm)
        temp = U[i,:]
        stretch = min(user_intensity[i], binge_time_cap)/binge_time_cap # \in [0,1]
        if sum(temp) > 0: # if L_1 norm > 0
            U[i,:] = stretch * (temp/sum(temp)) # can view this as stretch times angle
            user_wt[i] = 1 # recall: user_weight measures the usefulness of a user
        else: # if all-zero vector
            if display_flag == 'display':
                print("row", i, "is meaningless.")
            U[i,:] = np.zeros(num_col)+0.01 # still, perturb this row to avoid dividing by 0. Numpy can't compute 0*log(0)
            
    if display_flag == 'display':
        print("row", i, "is meaningless.")
        print("number of users inci to this card (i.e. num_lnb):", num_lnb)
        print("Binary reward vector:", rew_vec)
        print("User weights:", user_wt) # should be mostly 1
    return U        


def find_U(binge_time_cap, theta, display_flag):
    idx = 0 # need to know where to write in U

    for row in iem.iterrows(): # loop over all merged edges incident to this card
        ## compute user vector U
        user_temp = row[1]['user_id_hashed'] # get user id
        temp = user_stat[user_stat.user_id_hashed == user_temp] # search for this user in user_stat. Should be just one row
        if temp.shape[0] != 1: # if search result is not unique
            print("either not unique or not exist")
        else: #### HEART ####
            j = 0 # column index
            [num_merged_edge, cum_binge_time] = [0,0] # want to measure user intensity, this can be done by using user_stat
            for key in temp: # loop over all columns (categories)
                if j >= 1 and j <= num_col: # j=0 is the user_id column. skip this column
                    s = temp[key].iloc[0] # the list stored is a str of the form '[a,b,c]'
                    list_temp = remove_quote(s) # so we need to remove the quote. Now list_temp is a list floats 
                    if list_temp[0] >= 5: # Embedding rule: if num app >=5, then write the avg binge time into U
                        U[idx, j-1] = list_temp[1]/list_temp[0] # avg binge time (per app)
                    num_merged_edge += list_temp[0] # weighted app of this key
                    cum_binge_time += list_temp[1] # (merged) binge_time of this key
                j += 1

        if num_merged_edge != 0:
            user_intensity[idx] = cum_binge_time/num_merged_edge # binge time per card
        true_binge_time[idx] = row[1]['duration'] # true binge time
        if row[1]['duration'] > theta: # compute reward vector b
            rew_vec[idx] = 1
        if display_flag == 'display': # display. Can be turned off
            print("user id:", user_temp)
            print("cum. binge time:", round(cum_binge_time/10**3,3), "sec.") 
            print("num_merged_edge:", num_merged_edge)
            print("avg binge time per (merged) edge ('user_intensity'):", round(user_intensity[idx]/10**3,3), "sec.")
            print("details:")
            display(temp.iloc[0])
            print("*"*50)
        idx += 1

    for i in range(num_lnb):# remember to normalize each row of U (wrt L1 norm)
        temp = U[i,:]
        stretch = min(user_intensity[i], binge_time_cap)/binge_time_cap # \in [0,1]
        if sum(temp) > 0: # if L_1 norm > 0
            U[i,:] = stretch * (temp/sum(temp)) # can view this as stretch times angle
            user_wt[i] = 1 # recall: user_weight measures the usefulness of a user
        else: # if all-zero vector
            if display_flag == 'display':
                print("row", i, "is meaningless.")
            U[i,:] = np.zeros(num_col)+0.01 # still, perturb this row to avoid dividing by 0. Numpy can't compute 0*log(0)
            
    if display_flag == 'display':
        print("row", i, "is meaningless.")
        print("number of users inci to this card (i.e. num_lnb):", num_lnb)
        print("Binary reward vector:", rew_vec)
        print("User weights:", user_wt) # should be mostly 1
    return U


def log_likelihood(U,b,v,model_name,w): # compute the log likelihood
    # b = reward binary vec
    # U = user vec repn
    # v = card vec
    num_u = U.shape[0]
    product = 1
    ip = np.matmul(U, v)
    if model_name in ['model1','L_inf']:  # model: reward ~ Ber(<u,v>)
        for i in range(num_u):
            product *= (ip[i]** (w[i]*b[i]) ) * (1-ip[i])**((1-b[i])*w[i])
    if model_name == 'model2':  # model: reward ~ Ber(0.5 + 0.5 <u,v>)
        for i in range(num_u):        
            product *= (0.5+ 0.5*ip[i])** (b[i]*w[i]) * (0.5- 0.5*ip[i])**( (1-b[i])* w[i] ) 
    return np.log(product)


def merge_edges(binge_edge, name_temp, node_type, display_flag): # return the merged edges inci to a card, along with their binge times 
    # Recall that in binge_edge, there may be multiple edges between a user and a card. This function will merge those edges, and replace the meta-binge-time to be the max binge time over the edges we merged.
    # node_type: determines whether we merge the edges incident to a user or a card. Takes values 'card' or 'user'.
    # name_temp: The node whose incident edges are about to be merged by this function. Takes value either user_id_hashed or glance_id, depending on node_type.
    # display_flag: Determines whether display output's info. Takes values 'on' or 'off'. 
    # invoked other functions: none
    if node_type == 'card':
        card_temp = name_temp
        inci_edge = binge_edge[binge_edge.glance_id == card_temp] #edges incident to card_temp
        num = inci_edge.user_id_hashed.unique().shape[0]
        data = {'user_id_hashed':inci_edge.user_id_hashed.unique(),'duration':np.zeros(num)} # build a new df, each row per user
        iem = pd.DataFrame(data) # iem stands for 'inci_edge_merged'
        for row in inci_edge.iterrows(): # merge the edges with identical left nodes.
            user_temp = row[1][0] # get user id
            temp = iem[iem.user_id_hashed == user_temp] # search for the row of this user in iem
            idx = temp.index[0] # get the row index of this user in iem
            if row[1]['duration'] > iem.iloc[idx]['duration']: 
                iem.at[idx,'duration'] = row[1]['duration'] # the 'weight' of the merged edge is the max of all relevant edges
    if node_type == 'user':
        user_temp = name_temp
        inci_edge = binge_edge[binge_edge.user_id_hashed == user_temp] # edges incident to this user
        num = inci_edge.glance_id.unique().shape[0]
        data = {'glance_id':inci_edge.glance_id.unique(),'duration':np.zeros(num)} # build a new df, each row corres to a card
        iem = pd.DataFrame(data) # iem stands for inci_edge_merged
        for row in inci_edge.iterrows(): # merge the edges with identical left nodes.
            card_temp = row[1]['glance_id'] # get card id
            temp = iem[iem.glance_id == card_temp] # search for the row of this card in iem
            idx = temp.index[0] # get the row index of this card in iem
            if row[1]['duration'] > iem.iloc[idx]['duration']: 
                iem.at[idx,'duration'] = row[1]['duration']# the 'weight' of the merged edge is the max of all relevant edges
    if display_flag == 'display':
        print("each merged edge looks like:")
        display(iem.head())
        print("number of merged edges", iem.shape[0])
    return iem


def remove_quote(s):
    # input: a string that looks like '[a,b,c]'
    # out: a list
    s = s.replace('[','')
    s = s.replace(']','')
    temp1 = s.split(',')
    for i in range(len(temp1)):
        temp1[i] = float(temp1[i])
    return temp1