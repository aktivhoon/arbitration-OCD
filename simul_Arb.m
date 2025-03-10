function [data_out] = simul_Arb(param_in, data_in, mode)

%% create MAP
map_opt.transition_prob_seed=[0.7 0.3];  % init val, changed depending on block condition later
map_opt.reward_seed=[40 20 10 0];
[myMap N_state N_action N_transition]=Model_Map_Init2('sangwan2012b',map_opt);
[myMap_new N_state N_action N_transition]=Model_Map_Init2('sangwan2012c',map_opt);


%% create my arbitrator
myArbitrator=Bayesian_Arb_Init(N_state,N_action,N_transition);


%% create my RL
myState=Model_RL_Init(N_state,N_action,N_transition);


%% parameter plug-in
%{
param_in(1): myArbitrator.PE_tolerance_m1
param_in(2): myArbitrator.PE_tolerance_m2 / myArbitrator.m2_absPEestimate_lr
param_in(3): myArbitrator.A_12
param_in(x): myArbitrator.B_12 : based on A12
param_in(4): myArbitrator.A_21
param_in(x): myArbitrator.B_21 : based on A21
param_in(5): myArbitrator.tau_softmax/param_sarsa.tau/param_fwd.tau : better to fix at 0,2. This should be determined in a way that maintains softmax values in a reasonable scale. Otherwise, this will drive the fitness value!
param_in(6): % param_sarsa.alpha/param_fwd.alpha 0.01~0.2 to ensure a good "state_fwd.T" in phase 1
%}
param_fixed(1)=1; % 1: fwd-start, 2:sarsa-start
param_fixed(2)=1; % myArbitrator.p
param_fixed(3)=1e-1; % myArbitrator.Time_Step : time constant (1e0 (fast) ~ 1e-2 (slow))
param_sarsa.gamma=1.0; % fixed - not actual parameter
pop_id=1;
Sum_NegLogLik=0.0;

% arbitrator
myArbitrator.PE_tolerance_m1=param_in(pop_id,1);
% myArbitrator.PE_tolerance_m2=param_in(pop_id,2); % defines threshold for zero PE
myArbitrator.m2_absPEestimate_lr=param_in(pop_id,2); % defines the learning rate of RPE estimator
myArbitrator.Time_Step=param_fixed(3); % the smaller, the slower
switch mode.param_length
    case 6
        myArbitrator.A_12=param_in(pop_id,3);
        myArbitrator.B_12=log(myArbitrator.A_12/mode.boundary_12-1);
        myArbitrator.A_21=param_in(pop_id,4);
        myArbitrator.B_21=log(myArbitrator.A_21/mode.boundary_21-1);
        % SARSA
        param_sarsa.alpha=param_in(pop_id,6); % learning rate (0.1~0.2)
        % FWD
        param_fwd.alpha=param_in(pop_id,6);
        % Softmax parameter for all models
        myArbitrator.tau_softmax=param_in(pop_id,5); % use the same value as sarsa/fwd
        param_sarsa.tau=param_in(pop_id,5);
        param_fwd.tau=param_in(pop_id,5);
end

myArbitrator.ind_active_model=param_fixed(1);
if(myArbitrator.ind_active_model==1)
    myArbitrator.m1_prob_prev=0.7;
    myArbitrator.m2_prob_prev=1-myArbitrator.m1_prob_prev;
else
    myArbitrator.m1_prob_prev=0.3;
    myArbitrator.m2_prob_prev=1-myArbitrator.m1_prob_prev;
end
% non-linear weight : p
myArbitrator.p=param_fixed(2);

% arbitrator functioning mode
myArbitrator.opt_ArbModel=mode.opt_ArbModel;


%% Simulation
tot_num_sbj=size(data_in,2);
for ll=1:1:tot_num_sbj
    for kk=1:1:mode.total_simul
        state_fwd=myState;      state_fwd.name='fwd';
        state_sarsa=myState;    state_sarsa.name='sarsa';
        if(data_in{1,ll}.map_type==1)
            map=myMap;  map0=myMap; map0_s=myMap;
        end
        if(data_in{1,ll}.map_type==2)
            map=myMap_new;  map0=myMap_new; map0_s=myMap_new;   % map0 for fwd learning, map0_s for sarsa learning
        end
        
        
        %% (1) phase 1 - pretraining
        num_max_trial0=size(data_in{1,ll}.HIST_behavior_info_pre{1,1},1);
        map0.epoch=kk;                map0_s.epoch=kk;
        map0.data=data_in{1,ll}.HIST_behavior_info_pre{1,1};  map0_s.data=data_in{1,ll}.HIST_behavior_info_pre{1,1};
        opt_state_space.use_data=mode.experience_sbj_events(1); % use subject data for state-transition            

        if(mode.experience_sbj_events(1)~=-1)
            % fwd learning
            i=0;  cond=1;
            while ((i<num_max_trial0)&&(cond))
                i=i+1;
                map0.trial=i;
                
                % set T_prob
                block_condition=data_in{1,ll}.HIST_block_condition_pre{1,1}(2,i);
                if(block_condition==1) % G
                    prob_seed_mat=[0.9 0.1];
                end
                if(block_condition==2) % G'
                    prob_seed_mat=[0.5 0.5];
                end
                if(block_condition==3) % H
                    prob_seed_mat=[0.5 0.5];
                end
                if(block_condition==4) % H'
                    prob_seed_mat=[0.9 0.1];
                end
                
                % T_prob encoding to the current map
                if(data_in{1,ll}.map_type==1)
                    map0.action(1,1).prob(1,[2 3])=prob_seed_mat;
                    map0.action(1,1).prob(2,[7 9])=prob_seed_mat;
                    map0.action(1,1).prob(3,[8 9])=prob_seed_mat;
                    map0.action(1,1).prob(4,[7 9])=prob_seed_mat;
                    map0.action(1,1).prob(5,[6 9])=prob_seed_mat;
                    map0.action(1,2).prob(1,[4 5])=prob_seed_mat;
                    map0.action(1,2).prob(2,[8 7])=prob_seed_mat;
                    map0.action(1,2).prob(3,[7 9])=prob_seed_mat;
                    map0.action(1,2).prob(4,[6 7])=prob_seed_mat;
                    map0.action(1,2).prob(5,[7 9])=prob_seed_mat;
                end
                if(data_in{1,ll}.map_type==2)
                    for mm=1:1:2
                        for nn=1:1:size(map0.connection_info{1,mm},1)
                            map0.action(1,mm).prob(nn,map0.connection_info{1,mm}(nn,:))=prob_seed_mat;
                        end
                        map0.action(1,mm).connection=double(map0.action(1,mm).prob&ones(N_state,N_state));
                    end
                end
                
                % initializing the state
                param_fwd0=param_fwd;   param_fwd0.alpha=0.15;
                [state_fwd map0]=StateClear(state_fwd,map0);
                while (~state_fwd.JobComplete)
                    % decision
                    if(mode.experience_sbj_events(1)==1)
                        state_fwd=Model_RL2(state_fwd, param_fwd0, map0, 'decision_behavior_data_save');
                    else
                        state_fwd=Model_RL2(state_fwd, param_fwd0, map0, 'decision_random');
                    end
                    % state transition
                    [state_fwd map0]=StateSpace_v1(state_fwd,map0,opt_state_space);  % map&state index ++
                    % 1. fwd model update
                    if state_fwd.SARSA(1) == 1
                        state_fwd.SARSA(3) = 0;
                    end
                    state_fwd=Model_RL2(state_fwd, param_fwd0, map0, 'fwd_update');
                end
            end

            % sarsa learning
            i=0;  cond=1;
            while ((i<num_max_trial0)&&(cond))
                i=i+1;
                map0_s.trial=i;
                
                % set T_prob
                block_condition=data_in{1,ll}.HIST_block_condition_pre{1,1}(2,i);
                if(block_condition==1) % G
                    prob_seed_mat=[0.9 0.1];
                end
                if(block_condition==2) % G'
                    prob_seed_mat=[0.5 0.5];
                end
                if(block_condition==3) % H
                    prob_seed_mat=[0.5 0.5];
                end
                if(block_condition==4) % H'
                    prob_seed_mat=[0.9 0.1];
                end
                
                % T_prob encoding to the current map
                if(data_in{1,ll}.map_type==1)
                    map0_s.action(1,1).prob(1,[2 3])=prob_seed_mat;
                    map0_s.action(1,1).prob(2,[7 9])=prob_seed_mat;
                    map0_s.action(1,1).prob(3,[8 9])=prob_seed_mat;
                    map0_s.action(1,1).prob(4,[7 9])=prob_seed_mat;
                    map0_s.action(1,1).prob(5,[6 9])=prob_seed_mat;
                    map0_s.action(1,2).prob(1,[4 5])=prob_seed_mat;
                    map0_s.action(1,2).prob(2,[8 7])=prob_seed_mat;
                    map0_s.action(1,2).prob(3,[7 9])=prob_seed_mat;
                    map0_s.action(1,2).prob(4,[6 7])=prob_seed_mat;
                    map0_s.action(1,2).prob(5,[7 9])=prob_seed_mat;
                end
                if(data_in{1,ll}.map_type==2)
                    for mm=1:1:2
                        for nn=1:1:size(map0_s.connection_info{1,mm},1)
                            map0_s.action(1,mm).prob(nn,map0_s.connection_info{1,mm}(nn,:))=prob_seed_mat;
                        end
                        map0_s.action(1,mm).connection=double(map0_s.action(1,mm).prob&ones(N_state,N_state));
                    end
                end
                
                % initializing the state
                param_sarsa0=param_sarsa;   param_sarsa0.alpha=param_fwd0.alpha*1.2;
                [state_sarsa map0_s]=StateClear(state_sarsa,map0_s);
                while (~state_sarsa.JobComplete)
                    % 0. current action selection : (s,a) - using arbitrator's Q-value
                    if(mode.experience_sbj_events(1)==1)
                        state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'decision_behavior_data_save');
                    else
                        state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'decision_random');
                    end
                    % 1. sarsa state update (get reward and next state) : (r,s')
                    [state_sarsa map0_s]=StateSpace_v1(state_sarsa,map0_s,opt_state_space); % map&state index ++
                    % 1. sarsa next action selection : (s',a') - if s' is terminal, then no decision
                    state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'decision_hypo');
                    % 1. sarsa model upate
                    if state_sarsa.SARSA(1) == 1
                        state_sarsa.SARSA(3) = 0;
                    end
                    state_sarsa=Model_RL2(state_sarsa, param_sarsa0, map0_s, 'sarsa_update');
                end
            end
            
            % save current inital configuration
            data_in{1,ll}.init_state_fwd=state_fwd;
            data_in{1,ll}.init_state_sarsa=state_sarsa;
        else  % retrieve saved configuration
            state_fwd.Q=data_in{1,ll}.init_state_fwd.Q;
            state_sarsa.Q=data_in{1,ll}.init_state_sarsa.Q;
        end
        
        
        %% (2) phase 2 - main training
        num_max_session=size(data_in{1,ll}.HIST_behavior_info,2);
        cond=1;
        myArbitrator_top=myArbitrator;
        mode_data_prev=6;
        sbj = cell(num_max_session,1);
        for ind_sess=1:1:num_max_session
            % enter each session data into map
            i=0;
            num_max_trial=size(data_in{1,ll}.HIST_behavior_info{1,ind_sess},1);
            map.epoch=kk;
            map.data=data_in{1,ll}.HIST_behavior_info{1,ind_sess};
            
            while ((i<num_max_trial)&&(cond))
                i=i+1;
                map.trial=i;
                
                % mapping state-transition prob by block conditions
                block_condition=data_in{1,ll}.HIST_block_condition{1,ind_sess}(2,i);
                % set T_prob
                if((block_condition==1) || (block_condition==4)) % low uncertainty conditions
                    prob_seed_mat=[0.9 0.1];
                elseif((block_condition==2) || (block_condition==3)) % high uncertainty conditions
                    prob_seed_mat=[0.5 0.5];
                end
                % T_prob encoding to the current map
                if(data_in{1,ll}.map_type==2)
                    for mm=1:1:2
                        for nn=1:1:size(map.connection_info{1,mm},1)
                            map.action(1,mm).prob(nn,map.connection_info{1,mm}(nn,:))=prob_seed_mat;
                        end
                        map.action(1,mm).connection=double(map.action(1,mm).prob&ones(N_state,N_state));
                    end
                end
                
                % initializing the state
                [state_fwd map]=StateClear(state_fwd,map);
                [state_sarsa map]=StateClear(state_sarsa,map);
                
                % read a mode from data
                mode_data=map.data(map.trial,18);
                
                % mode change at every goal change
                myArbitrator_top.backward_flag=0;
                if((mode_data_prev~=mode_data) && (mode.USE_BWDupdate_of_FWDmodel==1))
                    myArbitrator_top.backward_flag=1;         % after backward update, should set it to 0.
                    if((mode_data_prev~=-1)&&(mode_data==-1)) % goal mode -> habitual mode
                        myArbitrator_top.backward_flag=0;
                    end
                end
                if(mode_data_prev~=mode_data)
                    myArbitrator_top.ind_active_model=1;  % switching the mode
                    myArbitrator_top.m1_prob_prev=0.8;    % changing the choice prob accordingly
                    myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                    if((mode_data_prev~=-1)&&(mode_data==-1))  % goal mode -> habitual mode
                        myArbitrator_top.ind_active_model=2;   % switching the mode
                        myArbitrator_top.m1_prob_prev=0.2;     % changing the choice prob accordingly
                        myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                    end
                end
                
                % forward only
                if(mode.USE_FWDSARSA_ONLY==1)
                    myArbitrator_top.ind_active_model=1;
                    myArbitrator_top.m1_prob_prev=0.99;
                    myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                    myArbitrator_top.Time_Step=1e-20;
                end
                % sarsa only
                if(mode.USE_FWDSARSA_ONLY==2)
                    myArbitrator_top.ind_active_model=2;
                    myArbitrator_top.m1_prob_prev=0.01;
                    myArbitrator_top.m1_prob=myArbitrator_top.m1_prob_prev;
                    myArbitrator_top.Time_Step=1e-20;
                end
                myArbitrator_top.m2_prob=1-myArbitrator_top.m1_prob;

                opt_state_space.use_data=mode.experience_sbj_events(2); % use subject data for state-transition
                while (((myArbitrator_top.ind_active_model==1)&&(~map.JobComplete))||((myArbitrator_top.ind_active_model==2)&&(~map.JobComplete)))
                    % index synchronization
                    state_fwd.index=map.index;
                    state_sarsa.index=map.index;

                    %% fwd mode: backward update of fwd model
                    % (1) revaluation
                    if(mode_data==-1) % reevaluation for habitual mode
                        mode_data_mat=[6 7 8 9];
                    else              % reevaluation for goal mode
                        mode_data_mat=mode_data;
                    end
                    map.reward=zeros(N_state,1);
                    map.reward(mode_data_mat)=map.reward_save(mode_data_mat);
                    
                    % (2) backward update of the fwd model
                    if((mode.USE_BWDupdate_of_FWDmodel==1) && (myArbitrator_top.backward_flag==1))
                        state_fwd=Model_RL2(state_fwd, param_fwd, map, 'bwd_update');
                    end
                    
                    
                    %% Compute negative log-likelihood : evaluate the arbitrator softmax using sbj's state,action
                    state_data=map.data(map.trial,3+map.index);
                    action_chosen=map.data(map.trial,6+map.index); % s, a pair
                    % compute real Q-value by merging two Q (fwd & sarsa)
                    myArbitrator_top.Q=...
                        ((myArbitrator_top.m1_prob*state_fwd.Q).^myArbitrator_top.p+...
                        (myArbitrator_top.m2_prob*state_sarsa.Q).^myArbitrator_top.p).^(1/myArbitrator_top.p);
                    myArbitrator_top.Q_old=myArbitrator_top.Q;
                    var_exp=exp(myArbitrator_top.tau_softmax*myArbitrator_top.Q(state_data,:)); % (N_actionx1)
                    Lik = var_exp/sum(var_exp);
                    Lik = Lik(action_chosen);
                    eval_num=log(Lik);
                    Sum_NegLogLik=Sum_NegLogLik-eval_num/(mode.total_simul);


                    %% main computation
                    % simultaneous update of both fwd and sarsa models
                    
                    % fwd
                    if(myArbitrator_top.ind_active_model==1)
                        % 0. current action selection : (s,a) - using arbitrator's Q-value
                        if(mode.experience_sbj_events(2)==1)
                            state_fwd=Model_RL2(state_fwd, param_fwd, map, 'decision_behavior_data_save');
                        else
                            state_fwd=Model_RL2(state_fwd, param_fwd, map, 'decision_arbitrator', myArbitrator_top, state_sarsa);
                        end
                        % 1. fwd state update (get reward and next state) : (r,s')
                        [state_fwd map]=StateSpace_v1(state_fwd,map,opt_state_space); % map&state index ++
                        state_sarsa.index=state_fwd.index;
                        % 1. fwd model update
                        if state_fwd.SARSA(1) == 1
                            state_fwd.SARSA(3) = 0;
                        end
                        state_fwd=Model_RL2(state_fwd, param_fwd, map, 'fwd_update');
                        % 2. state synchronization
                        state_sarsa.state_history(state_fwd.index)=state_fwd.state_history(state_fwd.index);
                        state_sarsa.SARSA=state_fwd.SARSA;
                        % 3. sarsa next action selection : (s',a') - if s' is terminal, then no decision
                        state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'decision_hypo');
                        % 3. sarsa model upate
                        if state_sarsa.SARSA(1) == 1
                            state_sarsa.SARSA(3) = 0;
                        end
                        state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'sarsa_update');
                        % history synchronization
                        myArbitrator_top.state_history=state_fwd.state_history;
                        myArbitrator_top.reward_history=state_fwd.reward_history;
                    end
                    
                    % sarsa
                    if(myArbitrator_top.ind_active_model==2)
                        % 0. current action selection : (s,a) - using arbitrator's Q-value                            
                        if(mode.experience_sbj_events(2)==1)
                            state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'decision_behavior_data_save');
                        else
                            state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'decision_arbitrator', myArbitrator_top, state_fwd);
                        end
                        % 1. sarsa state update (get reward and next state) : (r,s')
                        [state_sarsa map]=StateSpace_v1(state_sarsa,map,opt_state_space); % map&state index ++
                        state_fwd.index=state_sarsa.index;
                        % 1. sarsa next action selection : (s',a') - if s' is terminal, then no decision
                        state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'decision_hypo');
                        % 1. sarsa model upate
                        if state_sarsa.SARSA(1) == 1
                            state_sarsa.SARSA(3) = 0;
                        end
                        state_sarsa=Model_RL2(state_sarsa, param_sarsa, map, 'sarsa_update');
                        % 2. state synchronization
                        state_fwd.state_history(state_sarsa.index)=state_sarsa.state_history(state_sarsa.index);
                        state_fwd.SARSA=state_sarsa.SARSA;
                        if state_fwd.SARSA(1) == 1
                            state_fwd.SARSA(3) = 0;
                        end
                        % 3. fwd model update
                        state_fwd=Model_RL2(state_fwd, param_fwd, map, 'fwd_update');
                        % history synchronization
                        myArbitrator_top.state_history=state_sarsa.state_history;
                        myArbitrator_top.reward_history=state_sarsa.reward_history;
                    end
                    
                    % [ARBITRATOR] 
                    [myArbitrator_top, state_fwd, state_sarsa]=Bayesian_Arb(myArbitrator_top, state_fwd, state_sarsa, map);
                end
                mode_data_prev=mode_data;
            end
        end
    end        
end


%% fitness value
Sum_NegLogLik_val(pop_id)=Sum_NegLogLik;


%% returns
if(mode.out==1)
    data_out=Sum_NegLogLik_val(pop_id,1);
end
if(mode.out==2)
    data_out= sbj;
end

end








