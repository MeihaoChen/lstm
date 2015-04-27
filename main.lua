--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(2)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
stringx = require('pl.stringx')
require 'io'
require('nngraph')
require('base')
require('torch')
ptb = require('data')
-- Train 1 day and gives 82 perplexity.
local params = {batch_size=20,
                seq_length=50,
                layers=2,
                decay=1.15,
                rnn_size=200,
                dropout=0.5,
                init_weight=0.05,
                lr=1,
                vocab_size=50,
                max_epoch=14,
                max_max_epoch=20,
                max_grad_norm=10}

--[[
-- Trains 1h and gives test 115 perplexity.
local params = {batch_size=100,
                seq_length=50,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=50,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}
]]--
function transfer_data(x)
  return x:cuda()
end

--local state_train, state_valid, state_test
model = {}
--local paramx, paramdx

function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s),nn.Identity()(pred)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)

  return transfer_data(module)
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  model.pred = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    model.pred[j] = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i], model.pred[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local dpred = transfer_data(torch.zeros(params.batch_size,params.vocab_size))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds, dpred})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(5.6*perp / len)))
  g_enable_dropout(model.rnns)
end

function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, model.s[1], model.pred[i] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(5.6*perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

function train()
    words_per_step = params.seq_length * params.batch_size
    epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)

    while epoch < params.max_max_epoch do
     perp = fp(state_train)
     if perps == nil then
       perps = torch.zeros(epoch_size):add(perp)
     end
     perps[step % epoch_size + 1] = perp
     step = step + 1
     bp(state_train)
     total_cases = total_cases + params.seq_length * params.batch_size
     epoch = step / epoch_size
     if step % torch.round(epoch_size / 10) == 10 then
       wps = torch.floor(total_cases / torch.toc(start_time))
       since_beginning = g_d(torch.toc(beginning_time) / 60)
       print('epoch = ' .. g_f3(epoch) ..
         ', train perp. = ' .. g_f3(torch.exp(5.6*perps:mean())) ..
         ', wps = ' .. wps ..
         ', dw:norm() = ' .. g_f3(model.norm_dw) ..
         ', lr = ' ..  g_f3(params.lr) ..
         ', since beginning = ' .. since_beginning .. ' mins.')
     end
     if step % epoch_size == 0 then
       run_valid()
       if epoch > params.max_epoch then
           params.lr = params.lr / params.decay
       end
     end
     if step % 33 == 0 then
       cutorch.synchronize()
       collectgarbage()
     end
    end
    print('==> saving model')
    torch.save('/scratch/mc5283/a4/char_pred_medium_model.net', model)
end

function table_invert(t)
    local s={}
    for k,v in pairs(t) do
        s[v]=k
    end
    return s
end

function generating_sequence()
    path = '/scratch/mc5283/a4/word_pred_model.net'
    model = torch.load(path)
    state_train = {data=transfer_data(ptb.traindataset(params.batch_size))} 
    map = ptb.vocab_map
    reverse_map = table_invert(map)
    function readline()
        local line = io.read('*line')
        if line == nil then error({code='EOF'}) end
        line = stringx.split(line)
        if tonumber(line[1]) == nil then error({code="on table_invert(t)it"}) end
        input_len = #line - 1
        len = tonumber(line[1])
        data = torch.zeros(len, params.batch_size)
        for i = 2,#line do
            if map[line[i]] == nil then error({code="vocab", word = line[i]}) end
            x = torch.Tensor({map[line[i]]})
            data[i-1] = x:resize(x:size(1), 1):expand(x:size(1), params.batch_size) 
        end
        return data
    end
    while true do
        print('Query: len word1 word2 etc')
        local ok, line = pcall(readline)
        if not ok then 
            if line.code == 'EOF' then
                break
            elseif line.code == 'vocab' then
                print('Word not in vocabulary', line.word)
            elseif line.code == 'init' then
                print(line)
                print('Failed, try again')
            end
        else
            query = {}
            pred = {}
            query.data = data:cuda()
            reset_state(query)
            g_disable_dropout(model.rnns)
            local perp = 0
            local len = query.data:size(1)
            g_replace_table(model.s[0], model.start_s)
            for i = 1, (len - 1) do
                local x = query.data[i]
                local y = query.data[i + 1]
                local s = model.s[i - 1]
                perp_tmp, model.s[1], pred[i] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
                prob,ind = pred[i][1]:max(1)
                if i >= input_len then
                    query.data[i+1] = ind:resize(ind:size(1), 1):expand(ind:size(1), params.batch_size)
                end
                perp = perp + perp_tmp[1]
                g_replace_table(model.s[0], model.s[1])
            end
            words = {}
            for i = 1,len do
                word = query.data[i][1]
                words[i] = reverse_map[word]
            end
            print(table.concat(words," "))
        end
    end
end
function evaluation()
    filename = '/scratch/mc5283/a4/char_pred_large_model.net'
    model = torch.load(filename)
    state_train = {data=transferdata(ptb.traindataset(params.batch_size))}
    char_map = ptb.vocab_map
    print('OK GO')
    io.flush()
    function readline()
        local line = io.read('*line')
        if line == nil then error({code='EOF'}) end
        if char_map[line] == nil then error({code="vocab", word = line}) end
        input = torch.Tensor({char_map[line]})
        input = input:resize(input:size(1),1):expand(input:size(1), params.batch_size)
        return input
    end
    while true do
        local ok,line = pcall(readline)
        if not ok then
            if line.code == 'EOF' then
                print('word not in vocabulary')
                io.flush()
            elseif line.code == 'vocab' then
                print('word not in vocabulary')
                io.flush()
            end
        else
            test = {}
            test.data = input:cuda()
            reset_state(test)
            g_disable_dropout(model.rnns)
            local perp = 0
            g_replace_table(model.s[0], model.start_s)
            local x = test.data[1]
            local y = torch.zeros(params.batch_size):cuda()
            local s = model.s[0]
            perp_tmp, model.s[1], pred = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
            prob = pred[1]
            out = {}
            for i = 1, params.seq_length do
                out[i] = prob[i]
            end
            print(table.concat(out," "))
            io.flush()
        end
    end
end

function main()

    if not opt then
        print '==> processing options'
        cmd = torch.CmdLine()
        cmd:text('Options:')
        cmd:option('-mode', 'evaluate', 'train, query or evaluate')
        cmd:text()
        opt = cmd:parse(arg or {})
    end

    if opt.mode == 'train' then
        g_init_gpu({})
        state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
        state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
        --state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
        print("Network parameters:")
        print(params)
        local states = {state_train, state_valid, state_test}
        for _, state in pairs(states) do
         reset_state(state)
        end
        
        print('Setting up')
        setup()
        step = 0
        epoch = 0
        total_cases = 0
        beginning_time = torch.tic()
        start_time = torch.tic()
        print("Starting training.")
        train()
    
    elseif opt.mode == 'query' then
        generating_sequence()

    elseif opt.mode == 'evaluate' then
        evaluation()
    end
end

main()

