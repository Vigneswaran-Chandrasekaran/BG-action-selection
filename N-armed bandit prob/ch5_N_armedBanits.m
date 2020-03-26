%% Basal Ganglia as an engine for exploration

n_slots = 1000;
underlying_prob_dist = rand(1, n_slots);
initial_estimate = 0.7;

Stimulus = rand(1, n_slots);

W_g = rand(n_slots);
W_n = rand(n_slots);

dopmaine_tonic = 1.4;

n_episodes = 1000;

temp_diff_err = zeros(n_episodes, n_slots);
gradient_clipp = zeros(n_episodes, n_slots);
delta_Stimulus = zeros(n_episodes, n_slots);
reward_awarded = zeros(1, n_episodes);

tot_reward_awarded = 0;

act_Go = zeros(n_episodes, n_slots);
act_NGo = zeros(n_episodes, n_slots);
act_GPe = zeros(n_episodes, n_slots);
act_GPi = zeros(n_episodes, n_slots);
act_Th = zeros(n_episodes, n_slots);
act_STN = zeros(n_episodes, n_slots);

inhb = -1;

Wit = diag(inhb * rand(1,n_slots));
Wne = diag(inhb * rand(1,n_slots));
Wgi = diag(inhb * rand(1,n_slots));

W_e_STN = diag(inhb * rand(1,n_slots));
W_STN_e = diag(rand(1,n_slots));
W_STN_i = diag(rand(1,n_slots));

W_STN = rand(n_slots);
W_e = rand(n_slots);

alphaa = 1.4;
betaa = -1;

temp_err_mon = zeros(n_episodes - 1,1);
reward_mon = zeros(n_episodes - 1,1);

for t = 1 : n_episodes - 1
    disp(t)
    reward_vector = zeros(1, length(underlying_prob_dist));
    
    for k = 1 : length(underlying_prob_dist)
        if rand() < underlying_prob_dist(k)
            reward_vector(k) = 1;
        end
    end
    
    max_possible_reward = sum(reward_vector);
    act_Go(t,:) = sigmoid(Stimulus * W_g);
    act_NGo(t,:) = sigmoid(Stimulus * W_n);
    
    act_GPe(t,:) = sigmoid((act_NGo(t,:) * Wne) + (act_STN(t,:) * W_STN_e));
    act_STN(t,:) = sigmoid(act_GPe(t,:) * W_e_STN);
    act_GPi(t,:) = sigmoid((act_Go(t,:) * Wgi) + (act_STN(t,:) * W_STN_i));

    act_Th(t,:) = sigmoid(act_GPi(t,:) * Wit);

    winning_threshold = 0.5;

    winning_slots = find( act_Th(t,:) > winning_threshold);

    temp_diff_err(t,:) = reward_vector - act_Th(t,:);
    temp_diff_err(temp_diff_err < 0) = 0;

    for k = 1 : length(winning_slots)
        reward_awarded(t) = reward_awarded(t) + reward_vector(k);
    end
    
    tot_reward_awarded = tot_reward_awarded + reward_awarded(t);

    if (t == 1)
        gradient_clipp(t,:) = 0;
    else
        gradient_clipp(t,:) = act_Th(t,:) - act_Th(t-1,:);
    end

    lr = 10/(t+1);

    delta_Wg = lr * temp_diff_err(t,:)' * Stimulus;
    delta_Wn = lr * temp_diff_err(t,:)' * Stimulus;

    W_g = W_g - delta_Wg;
    W_n = W_n -delta_Wn;

    D_hi = 0.1;
    D_low = -0.3;

    
    Stimulus = Stimulus + delta_Stimulus(t+1,:);
    temp_err_mon(t) = sum(temp_diff_err(t));
    reward_mon(t) = reward_awarded(t) / max_possible_reward;
end

plot(reward_mon)
plot(temp_err_mon)
