library(ggplot2)
library(cowplot)





gg1 <- ggplot(train_eu, aes(x=group, y=bias)) +
  geom_violin(draw_quantiles =0.5) +
  labs(title="Leave-US-Out Cross-validation",x="Continent", y = "Bias")+ theme(plot.title = element_text(face="bold"))

gg2 <- ggplot(train_us, aes(x=group, y=bias)) +
  geom_violin(draw_quantiles =0.5) +
  labs(title="Leave-Europe-Out Cross-validation",x="Continent", y = "Bias")+ theme(plot.title = element_text(face="bold"))



plot_grid(gg1   , gg2  , labels = c('a', 'b'), nrow=1)
