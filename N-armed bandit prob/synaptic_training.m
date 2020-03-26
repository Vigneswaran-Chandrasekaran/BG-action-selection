%% Synapses training for action selection in Basal Ganglia
% References: 
% 1. https://doi.org/10.1007/978-981-10-8494-2_5
% 2. https://senselab.med.yale.edu/modeldb/ShowModel?model=239530&file=/UrsinoBaston2018/Addestramento_sinapsi.m#tabs-1 

%% Define basal stimuli

% number of stimuli for the network
N_stimuli = 4;

% Stimulus: 1
Stimulus_1(1) = 1.0;    % corresponding stimulus is initiated with 1 and others with 0
Stimulus_1(2:4) = 0.0;
Stimulus_1 = Stimulus_1';

% Stimulus: 2
Stimulus_2(1) = 0.0;
Stimulus_2(2) = 1.0;
Stimulus_2(3:4) = 0.0;
Stimulus_2 = Stimulus_2';

% Respective winners
Correct_winner = [1, 2];

%% Define initial synapses
% Four types of synapses are defined: 
% 1: Cortex to GO
% 2: Stimulus to GO
% 3: Cortex to NOGO
% 4: Stimulus to NOGO

N_channels = 4;

Synapse_cortex_GO = 0.5 * diag(ones(N_channels,1));
Synapse_cortex_NOGO = 0.5 * diag(ones(N_channels,1));
Synapse_stimuli_GO = 0.5 * diag(ones(N_channels,1));
Synapse_stimuli_NOGO = 0.5 * diag(ones(N_channels,1));

% set as mentioned in Ref: 2
Synapse_cortex_GO(3,3) = 0;
Synapse_cortex_GO(4,4) = 0;
Synapse_cortex_NOGO(3,3) = 0;
Synapse_cortex_NOGO(4,4) = 0;
Synapse_stimuli_GO(1,2) = 0.5;
Synapse_stimuli_GO(2,1) = 0.5;
Synapse_stimuli_NOGO(1,2) = 0.5;
Synapse_stimuli_NOGO(2,1) = 0.5;

%% Defining important entities for training
N_epochs = 10;

% Dopaminergic input
Dopamine_tonic = 1.2;

% Possible action's reward and punishment
Action_1_reward = 3;
Action_1_punishment = 5;

Action_2_reward = 2;
Action_2_punishment = 4;

% Synapses monitor at each epoch
Synapse_cortex_GO_monitor = zeros(N_channels, N_channels, N_epochs);
Synapse_cortex_NOGO_monitor = zeros(N_channels, N_channels, N_epochs);
Synapse_stimuli_GO_monitor = zeros(N_channels, N_channels, N_epochs);
Synapse_stimuli_NOGO_monitor = zeros(N_channels, N_channels, N_epochs);

% Initialize first time step with defined values
Synapse_cortex_GO_monitor(:,:,1) = Synapse_cortex_GO;
Synapse_cortex_NOGO_monitor(:,:,1) = Synapse_cortex_NOGO;
Synapse_stimuli_GO_monitor(:,:,1) = Synapse_stimuli_GO;
Synapse_stimuli_NOGO_monitor(:,:,1) = Synapse_stimuli_NOGO;

% Monitor reward, punishment and no_response at each epoch
Epoch_reward = zeros(N_epochs,1);
Epoch_punishment = zeros(N_epochs,1);
Epoch_noresponse = zeros(N_epochs,1);

% Monitor stimulus at each epoch
Epoch_S = zeros(2, N_epochs);
Epoch_S1 = zeros(2, N_epochs);
Epoch_S2 = zeros(2, N_epochs);

%% Start training
for i = 1:N_epochs

    Synapse_cortex_GO = squeeze(Synapse_cortex_GO_monitor(:,:,i));
    Synapse_cortex_NOGO = squeeze(Synapse_cortex_NOGO_monitor(:,:,i));
    Synapse_stimuli_GO = squeeze(Synapse_stimuli_GO_monitor(:,:,i));
    Synapse_stimuli_NOGO = squeeze(Synapse_stimuli_NOGO_monitor(:,:,i));

    Stimulus_choose = rem(i,2);
    noise = 0 * randn(2,1);      % TODO: remove 0

    if Stimulus_choose == 1
        Current_stimulus = Stimulus_1;
        Current_winner = Correct_winner(1);
    else
        Current_stimulus = Stimulus_2;
        Current_winner = Correct_winner(2);

    Current_stimulus(1) = Current_stimulus(1) + noise(1);
    Current_stimulus(2) = Current_stimulus(2) + noise(2);
    end

    % Quantize the resultant
    Current_stimulus(Current_stimulus > 1) = 1;
    Current_stimulus(Current_stimulus < 0) = 0;
    Current_stimulus(3:4) = 0;

    %%% BG_mechanism function

    if reward == 1
        Epoch_reward = 1;
    elseif reward == -1
        Epoch_punishment = 1;
    else
        Epoch_noresponse = 1;
    end

    if Stimulus_choose == 1
        Epoch_S1(1,i) = Current_stimulus(1);
        Epoch_S1(2,i) = Current_stimulus(2);
    
    elseif Stimulus_choose == 2
        Epoch_S2(1,i) = Current_stimulus(1);
        Epoch_S2(2,i) = Current_stimulus(2);
    end

    Synapse_stimuli_GO_monitor(:,:,i+1) = Synapse_stimuli_GO_POST;
    Synapse_stimuli_NOGO_monitor(:,:,i+1) = Synapse_stimuli_NOGO_POST;
    Synapse_cortex_GO_monitor(:,:,i+1) = Synapse_cortex_GO_POST;
    Synapse_cortex_NOGO_monitor(:,:,i+1) = Synapse_cortex_NOGO_POST;

end







