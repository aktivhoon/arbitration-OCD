function [] = batch_pcs(sis,MAXD)
    
    %% prep
    t = clock;
    LIST_SBJ ={'Subj001_AAA', 'Subj002_BBB', 'Subj003_CCC'};
    list_sbj = {LIST_SBJ{sis}};
    path_local = '/PATH/TO/LocalDIR/';
    
    
    %% param_in
    %{
    param1. zero prediction error threshold of SPE
    param2. learning rate for the estimate of absolute RPE
    param3. the amplitude of a transition rate function (mb->mf)
    param4. the amplitude fo a transition rate function (mf->mb)
    param5. inverse softmax temparature
    param6. learning rate of RL
    careful setting of the boundary to address overfitting
    %}
    mode.param_BoundL = [0.3 0.1  0.02 0.02 0.01 0.05];
    mode.param_BoundU = [0.7 0.35 20   20   0.5  0.2];    
    param_init= zeros(1,6);
    for i = 1 : 6
        rand(floor(mod(sum(clock*10),10000)));
        param_init(i) = rand  * (mode.param_BoundU(i) - mode.param_BoundL(i)) + mode.param_BoundL(i);
    end
    

    %% mode
    mode.param_length = size(param_init,2);
    mode.total_simul = 10;
    mode.experience_sbj_events = [1 1];  % ones(1,2); by subjs' exp (1), by model pred (0), use saved setting (-1)
    mode.USE_FWDSARSA_ONLY = 0;          % 0: arbitration, 1: use fwd only, 2: use sarsa only
    mode.USE_BWDupdate_of_FWDmodel = 1;  % 1: use the backward update for fwd model, 0: do not use
    mode.out = 1;
    mode.opt_ArbModel = 0;    % 0: dynamical transition, 1: uncertainty ratio, 2: posterior mean ratio
    mode.boundary_12 = 0.01;  % boundary condition (beta) of MB->MF transition; fitted using an independent dataset
    mode.boundary_21 = 0.01;  % boundary condition (alpha) of MF->MB transition
    mode.max_iter = 200;
    maxd = MAXD;


    %% data_in
    maxi=size(list_sbj,2);
    PreBehav = {};
    PreBlck = {};
    MainBehav = {};
    MainBlck = {};
    SBJ = cell(1, maxi);
    
    for i = 1 : maxi
        % prac data
        TEMP_PRE=load([path_local 'PATH/TO/raw_behav/' list_sbj{i} '_pre_1.mat']);
        PreBehav{i}=TEMP_PRE.HIST_behavior_info{1,1};
        PreBlck{i}=TEMP_PRE.HIST_block_condition{1,1};
        
        % main data
        tt = dir([path_local 'PATH/TO/raw_behav']);
        tt = {tt.name};
        maxsess = sum(cell2mat(strfind(tt,[list_sbj{i} '_fmri_']))) - 1;
        for ii = 1 : maxsess
            TEMP_MAIN=load([path_local 'PATH/TO/raw_behav/' list_sbj{i} '_fmri_' num2str(ii) '.mat']);
            MainBehav{i,ii}=TEMP_MAIN.HIST_behavior_info0;
            MainBlck{i,ii}=TEMP_MAIN.HIST_block_condition{1,ii};
        end
    end
    mkdir([path_local 'PATH/TO/proc_behav/' list_sbj{i}]) 
    save([path_local 'PATH/TO/proc_behav/' list_sbj{i} '/F_DATA.mat'], 'PreBehav', 'PreBlck','MainBehav', 'MainBlck');
    disp('DATA STORING DONE!');

    SBJtot = cell(maxd,1);
    for d = 1 : maxd
        [SBJ] = optim_Arb(maxi,mode,PreBehav,PreBlck,MainBehav,MainBlck,list_sbj,param_init)
        SBJtot{d,1} = SBJ;
    end
    list_month={'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'};
    save([path_local 'PATH/TO/proc_behav/' list_sbj{1} '/SBJ_' list_sbj{1} '_' num2str(t(1)) list_month{t(2)} num2str(t(3)) '_' num2str(t(4)) num2str(t(5)) num2str(floor(t(6))) '.mat'], 'SBJtot');
end
