function [myArbitrator myState1 myState2]=Bayesian_Arb(myArbitrator, myState1, myState2, myMap)
    %% model option
    modelOpt = 2;  % 1: dualBayesArb, 2: mixedArb
    
    
    %% Preparation
    myArbitrator.m1_thr_PE=myArbitrator.PE_tolerance_m1*[-1 1];
    myArbitrator.m2_thr_PE=myArbitrator.PE_tolerance_m2*[-1 1];  % length = myArbitrator.K-1
    
    if(myArbitrator.ind_active_model==1)
        myState=myState1;
    else
        myState=myState2;
    end
    
    
    %% index, state_history, action_history synchronization
    % simply inherit because both should be in the same state
    myArbitrator.index=myMap.index-1;
    myArbitrator.state_history(myArbitrator.index)=myState.state_history(myArbitrator.index);
    myArbitrator.action_history(myArbitrator.index)=myState.action_history(myArbitrator.index);


    %% Hierarchical Bayesian Inference
    % MB model (m1)
    myArbitrator.T_current1=min(myArbitrator.T_current1+1,myArbitrator.T); % update # of accumulated events
    % (0) backup old values
    myArbitrator.m1_mean_old=myArbitrator.m1_mean;  myArbitrator.m1_var_old=myArbitrator.m1_var;    myArbitrator.m1_inv_Fano_old=myArbitrator.m1_inv_Fano;
    % (1) find the corresponding row
    [tmp ind_neg]=find((myArbitrator.m1_thr_PE-myState1.SPE_history(myArbitrator.index+1))<0); % [!!] index + 1, if not, SPE is 0 for every first step
    ind_update=length(ind_neg)+1; % 1:neg, 2:zero, 3:posPE
    % (2) update the current column(=1) in PE_history
    myArbitrator.m1_PE_history(:,2:end)=myArbitrator.m1_PE_history(:,1:end-1); % shift 1 column (toward past)
    myArbitrator.m1_PE_history(:,1)=zeros(myArbitrator.K,1); % empty the first column
    myArbitrator.m1_PE_history(ind_update,1)=1; % add the count 1 in the first column
    myArbitrator.m1_PE_num=myArbitrator.m1_PE_history*myArbitrator.discount_mat'; % compute discounted accumulated PE
    % (3) posterior mean & var
    sumK=sum(myArbitrator.m1_PE_num);
    sumK_excl=sumK-myArbitrator.m1_PE_num;
    myArbitrator.m1_mean=(1+myArbitrator.m1_PE_num)/(myArbitrator.K+sumK);
    myArbitrator.m1_var=((1+myArbitrator.m1_PE_num)/((myArbitrator.K+sumK)^2))/(myArbitrator.K+sumK+1).*(myArbitrator.K+sumK_excl-1);
    % Here, reliabilty is instead replaced with uncertainty
    myArbitrator.m1_inv_Fano=myArbitrator.m1_var./myArbitrator.m1_mean;
    
    % MF model (m2)
    if modelOpt == 1  % dualBayesArb
        myArbitrator.T_current2=min(myArbitrator.T_current2+1,myArbitrator.T); % update # of accumulated events
        % (0) backup old values
        myArbitrator.m2_mean_old=myArbitrator.m2_mean;  myArbitrator.m2_var_old=myArbitrator.m2_var;    myArbitrator.m2_inv_Fano_old=myArbitrator.m2_inv_Fano;
        % (1) find the corresponding row
        [tmp ind_neg]=find((myArbitrator.m2_thr_PE-myState2.RPE_history(myState2.index))<0);  % [!!] must be sarsa because it looks into RPE.
        ind_update=length(ind_neg)+1;
        % (2) update the current column(=1) in PE_history
        myArbitrator.m2_PE_history(:,2:end)=myArbitrator.m2_PE_history(:,1:end-1); % shift 1 column (toward past)
        myArbitrator.m2_PE_history(:,1)=zeros(myArbitrator.K,1); % empty the first column
        myArbitrator.m2_PE_history(ind_update,1)=1; % add the count 1 in the first column
        myArbitrator.m2_PE_num=myArbitrator.m2_PE_history*myArbitrator.discount_mat'; % compute discounted accumulated PE
        % (3) posterior mean & var
        sumK=sum(myArbitrator.m2_PE_num);
        sumK_excl=sumK-myArbitrator.m2_PE_num;
        myArbitrator.m2_mean=(1+myArbitrator.m2_PE_num)/(myArbitrator.K+sumK);
        myArbitrator.m2_var=((1+myArbitrator.m2_PE_num)/((myArbitrator.K+sumK)^2))/(myArbitrator.K+sumK+1).*(myArbitrator.K+sumK_excl-1);
%         myArbitrator.m2_inv_Fano=myArbitrator.m2_mean./myArbitrator.m2_var;
        myArbitrator.m2_inv_Fano=myArbitrator.m2_var./myArbitrator.m2_mean;
    elseif modelOpt == 2  % mixedArb
        myArbitrator.m2_absPEestimate=myArbitrator.m2_absPEestimate+myArbitrator.m2_absPEestimate_lr*(abs(myState2.RPE_history(myState2.index))-myArbitrator.m2_absPEestimate);
        % Here, reliabilty is instead replaced with uncertainty
        myArbitrator.m2_inv_Fano=[0; myArbitrator.m2_absPEestimate/40; 0];
        myArbitrator.m2_mean=myArbitrator.m2_inv_Fano; myArbitrator.m2_var=[0.1; (myArbitrator.m2_absPEestimate*0.01); 0.1];
    end
    

    %% Dynamic Arbitration
    myArbitrator.temp=0;    %[myArbitrator.ind_active_model; input/sum(input0)];
    
    input0=myArbitrator.m1_inv_Fano;
    input1= myArbitrator.m1_wgt'*myArbitrator.m1_inv_Fano;
    chi_mb = input1/sum(input0);
    % Use 1-chi_mb, which is the reliability (1-uncertainty)
    myArbitrator.transition_rate12=myArbitrator.A_12/(1+exp(myArbitrator.B_12*(1-chi_mb)));
    if modelOpt == 1
        input0=myArbitrator.m2_inv_Fano;
        input2=myArbitrator.m2_wgt'*myArbitrator.m2_inv_Fano;
    elseif modelOpt == 2
        input0=1;
        input2=myArbitrator.m2_wgt'*myArbitrator.m2_inv_Fano;
    end
    chi_mf = input2/sum(input0);
    % Use 1-chi_mf, which is the reliability (1-uncertainty)
    myArbitrator.transition_rate21=myArbitrator.A_21/(1+exp(myArbitrator.B_21*(1-chi_mf)));
    myArbitrator.transition_rate12_prev=myArbitrator.transition_rate12;
    myArbitrator.transition_rate21_prev=myArbitrator.transition_rate21;

    myArbitrator.Tau=1/(myArbitrator.transition_rate12+myArbitrator.transition_rate21); % alpha + beta term.
    myArbitrator.m1_prob_inf=myArbitrator.transition_rate21*myArbitrator.Tau; % transition_rate21 = alpha
    myArbitrator.m1_prob=myArbitrator.m1_prob_inf+(myArbitrator.m1_prob_prev-myArbitrator.m1_prob_inf)*exp((-1)*myArbitrator.Time_Step/myArbitrator.Tau);
    
    switch myArbitrator.opt_ArbModel
        case 0 % dynamical transition
        case 1 % uncertainty ratio
            myArbitrator.m1_prob=input1/(input1+input2);
        case 2 % posterior mean ratio
            myArbitrator.m1_prob=myArbitrator.m1_mean(2)/(myArbitrator.m1_mean(2)+myArbitrator.m2_mean(2));        
    end
    myArbitrator.m1_prob_prev=myArbitrator.m1_prob;
    myArbitrator.m2_prob=1-myArbitrator.m1_prob;
    
    
    %% choice of the model
    myArbitrator.ind_active_model_prev=myArbitrator.ind_active_model;
    if(myArbitrator.m1_prob>0.5)
        myArbitrator.ind_active_model=1;
        myArbitrator.num_m1_chosen=myArbitrator.num_m1_chosen+1;
    else
        myArbitrator.ind_active_model=2;    
        myArbitrator.num_m2_chosen=myArbitrator.num_m2_chosen+1;
    end
    
    %% Q-value computation and action choice
    myArbitrator.Q=...
        ((myArbitrator.m1_prob*myState1.Q).^myArbitrator.p+...
        (myArbitrator.m2_prob*myState2.Q).^myArbitrator.p).^(1/myArbitrator.p);
end