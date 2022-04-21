library(ggplot2)
library(cowplot)



dbf <- DBF
mf <- MF
gra <- GRA
enf <- ENF

gg1 <- ggplot(dbf, aes(x=group, y=bias)) +
  geom_violin(draw_quantiles =0.5) +
  labs(title="LGOCV trained on DBF",x="Vegetation Type", y = "Bias")+ theme(plot.title = element_text(face="bold"))

gg2 <- ggplot(mf, aes(x=group, y=bias)) +
  geom_violin(draw_quantiles =0.5) +
  labs(title="LGOCV trained on MF",x="Vegetation Type", y = "Bias")+ theme(plot.title = element_text(face="bold"))

gg3 <- ggplot(gra, aes(x=group, y=bias)) +
  geom_violin(draw_quantiles =0.5) +
  labs(title="LGOCV trained on GRA",x="Vegetation Type", y = "Bias")+ theme(plot.title = element_text(face="bold"))


gg4 <- ggplot(enf, aes(x=group, y=bias)) +
  geom_violin(draw_quantiles =0.5) +
  labs(title="LGOCV trained on ENF",x="Vegetation Type", y = "Bias")+ theme(plot.title = element_text(face="bold"))


plot_grid( gg1   , gg2  , gg3  ,gg4, labels = c('a', 'b', 'c', 'd'), nrow=2, ncol=2)

