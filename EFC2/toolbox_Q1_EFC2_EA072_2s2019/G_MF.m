function [G_MF] = G_MF(k_P,k_D,k_I)
G = tf([400],[1 30 200 0]);
C = tf([k_D k_P k_I],[1 0]);
CG = C*G;
G_MF = feedback(CG,1);
