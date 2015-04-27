require 'nngraph'
require 'nn'

h1 = nn.Identity()()
h2 = nn.Identity()()
h3 = nn.Linear(20,10)()

out = nn.CAddTable()({h1, nn.CMulTable()({h2,h3})})

gmod = nn.gModule({h1,h2,h3}, {out})

x1 = torch.rand(10)
x2 = torch.rand(10)
x3 = torch.rand(20)

gmod:forward({x1,x2,x3})

