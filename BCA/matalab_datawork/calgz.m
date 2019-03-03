function mY = calgz(ggz)

global yy by mles mlee

nypc = yy(mles:mlee); T = size(nypc,1);
Y     = log(nypc.*(1+ggz).^(by-mles+1))-log(nypc(by-mles+1))-log((1+ggz).^(0:T-1)');
mY = mean(Y);