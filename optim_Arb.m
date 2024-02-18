function [SBJ]=optim_Arb(maxi,mode,PreBehav,PreBlck,MainBehav,MainBlck,list_sbj,param_init)
    
    tt = clock;
    path_local = '/PATH/TO/LocalDIR/';

    for ppi = 1 : 6
        rng(floor(tt(6)*1000));
        param_init(ppi) = randi(20)*(mode.param_BoundU(ppi) - mode.param_BoundL(ppi))/20 + mode.param_BoundL(ppi);
    end
    mode.param_init_ori = param_init;
    
    for i = 1 : maxi
        fprintf('###   SUB_NUM: [%d / %d]\n',i,maxi);
        fprintf('### OPT_ITER : [%d]\n',mode.max_iter);
        % pre save
        data_in{1,1}.map_type = 2;  % sangwan2012c map, validated in Lee et al. 2014, Neuron
        data_in{1,1}.HIST_behavior_info_pre{1,1} = PreBehav{i};
        data_in{1,1}.HIST_block_condition_pre{1,1} = PreBlck{i};
        
        % main save
        tt = dir([path_local 'PATH/TO/raw_behav']);
        tt = {tt.name};
        maxsess = sum(cell2mat(strfind(tt,[list_sbj{i} '_fmri_']))) - 1;
        for ii = 1 : maxsess
            data_in{1,1}.HIST_behavior_info{1,ii} = MainBehav{i,ii};
            data_in{1,1}.HIST_block_condition{1,ii} = MainBlck{i,ii};
        end
        
        % optimization
        myFunc_bu = @(x) simul_Arb(x, data_in, mode);
        disp(['    ***** subject number : [' num2str(i) '], subject name : [' list_sbj{i} ']' ]);
        einstein= 1;
        howmany=0;

        while (einstein==1)
            try
                [model_BayesArb.param, model_BayesArb.val] = fminsearchbnd(myFunc_bu, param_init, mode.param_BoundL, mode.param_BoundU, optimset('Display', 'final','MaxIter',mode.max_iter));
                einstein= 0;
                clc;
            catch
                einstein=1;
                for ppi = 1 : 6
                    rng(floor(tt(6)*1000));
                    param_init(ppi) = randi(20)*(mode.param_BoundU(ppi) - mode.param_BoundL(ppi))/20  + mode.param_BoundL(ppi);
                end
                mode.param_init_ori = param_init;
                howmany= howmany + 1;
                disp([num2str(howmany) 'error(s)']);
                if howmany > 1000
                    einstein=0;
                end
            end
        end
        model_BayesArb.mode = mode;
        
        % save fitted params
        SBJ{1,i} = data_in{1,1};
        SBJ{1,i}.model_BayesArb = model_BayesArb;
    end
end