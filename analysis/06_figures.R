# Figures 3, 4b, 5 and 6 of the manuscript and Supplementary Figure S1 (regression scatterplots),
# drawn from the shared data in data/ only. Figure 4a (example trajectories) needs raw head positions
# and is not redrawn here.
# Group comparisons use the same tests as the text: Student's t for spatial coverage, object coverage and
# search success, Welch's t for backtracking duration. Error bars are standard errors of the mean.
# Run from the repository root or from analysis/. Outputs: figures/*.png
suppressMessages({library(ggplot2); library(patchwork); library(jsonlite)})
root <- if (dir.exists("data")) "." else ".."
out_dir <- file.path(root, "figures"); dir.create(out_dir, showWarnings = FALSE)
P <- read.csv(file.path(root, "data/participants.csv"))
S <- read.csv(file.path(root, "data/subject_measures.csv"))
E <- read.csv(file.path(root, "data/events.csv"))
PS <- fromJSON(file.path(root, "results/participant_stats.json"))
MAX_ENC <- 4

# adaptation: per participant, slope of the change in stopping time on encounter number (MS events, encounters 1-4)
ms <- E[E$event_type == "MS" & E$encounter <= MAX_ENC, ]
adapt <- do.call(rbind, lapply(split(ms, ms$code), function(g)
  if (length(unique(g$encounter)) >= 2) data.frame(code = g$code[1], adaptation = coef(lm(stop_diff ~ encounter, g))[[2]])))
D <- merge(merge(S, P[, c("code", "age")], by = "code"), adapt, by = "code", all.x = TRUE)
D$opt1 <- D$seen_cells / D$avail_cells
D$Group <- factor(ifelse(D$group == "child", "Children", "Adults"), levels = c("Children", "Adults"))
C <- D[D$group == "child", ]

TEAL <- "#00A08A"; TEAL_D <- "#006B5E"; TEAL_L <- "#004D40"; ORANGE <- "#F98400"; ORANGE_D <- "#CC6600"; GREY <- "gray50"
FILL <- c(Children = TEAL, Adults = ORANGE); PT <- c(Children = TEAL_D, Adults = ORANGE_D)
theme_eye <- function(size = 20) theme_minimal(base_size = size) + theme(
  text = element_text(family = "sans"), axis.text = element_text(size = size - 2), axis.title = element_text(size = size),
  plot.title = element_text(size = size + 2, face = "bold", hjust = 0.5), panel.grid = element_blank(),
  panel.border = element_rect(colour = "gray30", fill = NA, linewidth = 0.5),
  strip.text = element_text(size = size, face = "bold"), plot.tag = element_text(size = size + 12, face = "bold"))
lim_for <- function(v, step) c(0, ceiling((max(D[[v]], na.rm = TRUE) + 0.14 * step * 4) / step) * step)   # room above the data for the bracket
brk <- function(lim) { b <- pretty(c(0, lim[2] * 0.93)); b[b <= lim[2]] }   # axis labels stop below the bracket
stars <- function(p) if (p < .001) "***" else if (p < .01) "**" else if (p < .05) "*" else ""

bars <- function(v, ylab, lim, welch = FALSE, top = max(D[[v]], na.rm = TRUE)) {
  tt <- t.test(D[D$group == "adult", v], D[D$group == "child", v], var.equal = !welch)
  s <- data.frame(Group = factor(c("Children", "Adults"), levels = levels(D$Group)),
                  m = tapply(D[[v]], D$Group, mean)[c("Children", "Adults")],
                  se = tapply(D[[v]], D$Group, function(x) sd(x) / sqrt(length(x)))[c("Children", "Adults")])
  ys <- top + diff(lim) * 0.04   # bracket sits above the highest point; lim leaves room for it
  g <- ggplot() +
    geom_col(data = s, aes(Group, m, fill = Group), alpha = 0.7, width = 0.6) +
    geom_errorbar(data = s, aes(Group, ymin = m - se, ymax = m + se), width = 0.2, linewidth = 0.8) +
    geom_jitter(data = D, aes(Group, .data[[v]], colour = Group), width = 0.15, height = 0, size = 2, alpha = 0.7) +
    scale_fill_manual(values = FILL, guide = "none") + scale_colour_manual(values = PT, guide = "none") +
    scale_y_continuous(limits = lim, breaks = brk(lim), expand = expansion(mult = c(0, 0.02))) +
    labs(x = "Age Group", y = ylab) + theme_eye()
  if (tt$p.value < .05) g <- g +
    annotate("segment", x = 1, xend = 2, y = ys, yend = ys, linewidth = 0.6) +
    annotate("segment", x = c(1, 2), xend = c(1, 2), y = ys - diff(lim) * 0.012, yend = ys, linewidth = 0.6) +
    annotate("text", x = 1.5, y = ys - diff(lim) * 0.01, label = stars(tt$p.value), size = 13, fontface = "bold", vjust = 0)
  g
}
AGE_KEY <- c(spatial_coverage = "age_spatial_coverage", opt1 = "age_opt1", BT = "age_BT", success_proportion = "age_success")
age_scatter <- function(v, lim) {
  # line fitted after the outlier rule of the age regression reported in the text; excluded children as open circles
  am <- mean(D[D$group == "adult", v]); ex <- unlist(PS[[AGE_KEY[[v]]]]$excluded)
  X <- C; X$excluded <- X$code %in% ex
  ggplot(X, aes(age, .data[[v]])) +
    geom_hline(yintercept = am, colour = ORANGE, linewidth = 1.5, alpha = 0.9) +
    annotate("text", x = 6.55, y = am - diff(lim) * 0.045, label = "Adult Mean", colour = ORANGE, size = 5.6, fontface = "bold", hjust = 0) +
    geom_smooth(data = X[!X$excluded, ], method = "lm", formula = y ~ x, colour = TEAL_L, fill = TEAL, alpha = 0.2, linewidth = 1.3) +
    geom_point(data = X[!X$excluded, ], colour = TEAL, size = 3.6, alpha = 0.75) +
    geom_point(data = X[X$excluded, ], colour = TEAL_L, size = 3.6, shape = 21, fill = NA, stroke = 1.3) +
    scale_x_continuous(breaks = 7:14, limits = c(6.5, 14.5)) +
    scale_y_continuous(limits = lim, breaks = brk(lim), expand = expansion(mult = c(0, 0.02))) +
    labs(x = "Age (years)", y = NULL) + theme_eye()
}
save <- function(p, name, w, h) { ggsave(file.path(out_dir, name), p, width = w, height = h, dpi = 300, bg = "white"); cat("saved", name, "\n") }

# ---------------------------------------------------------------- Figure 3: spatial coverage and object coverage
lim_oc <- lim_for("opt1", 0.1)
lim_sc <- c(0, 1)
f3 <- (bars("spatial_coverage", "Spatial Coverage\n(Proportion of Grid Cells Visited)", lim_sc) | age_scatter("spatial_coverage", lim_sc)) /
      (bars("opt1", "Object Coverage\n(Share of Nearby Objects Looked At)", lim_oc) | age_scatter("opt1", lim_oc)) +
  plot_annotation(title = "Spatial and Object Coverage", tag_levels = list(c("a", "", "b", "")),
                  theme = theme(plot.title = element_text(size = 26, face = "bold", hjust = 0.5)))
save(f3, "fig3_global_exploration.png", 13, 12)

# ---------------------------------------------------------------- Figure 4b: backtracking duration
lim_bt <- lim_for("BT", 50)
f4b <- (bars("BT", "Total Backtracking Duration (seconds)", lim_bt, welch = TRUE) | age_scatter("BT", lim_bt)) +
  plot_annotation(tag_levels = list(c("b", "")))
save(f4b, "fig4b_backtracking_duration.png", 13, 6.3)

# ---------------------------------------------------------------- Figure 5: stopping after MS and control events, encounters 1-4
E4 <- E[E$encounter <= MAX_ENC, ]
E4$Group <- factor(ifelse(E4$group == "child", "Children", "Adults"), levels = c("Children", "Adults"))
E4$Event <- factor(ifelse(E4$event_type == "MS", "Multisensory Events", "Control Events"), levels = c("Multisensory Events", "Control Events"))
s5 <- aggregate(stop_diff ~ Group + Event + encounter, E4, function(x) c(m = mean(x), se = sd(x) / sqrt(length(x)), n = length(x)))
s5 <- cbind(s5[, 1:3], as.data.frame(s5$stop_diff))
s5$col <- ifelse(s5$Event == "Control Events", "Control", as.character(s5$Group))
f5 <- ggplot(s5, aes(encounter, m, colour = col, group = Event)) +
  geom_hline(yintercept = 0, colour = "gray80", linewidth = 0.5) +
  geom_errorbar(aes(ymin = m - se, ymax = m + se), width = 0.1, linewidth = 1.0, alpha = 0.85) +
  geom_line(linewidth = 1.8, alpha = 0.9) + geom_point(size = 5, alpha = 0.95) +
  facet_wrap(~ Group) +
  scale_colour_manual(values = c(Children = TEAL, Adults = ORANGE, Control = GREY),
                      breaks = c("Children", "Adults", "Control"), labels = c("Multisensory Events (children)", "Multisensory Events (adults)", "Control Events")) +
  scale_x_continuous(breaks = 1:MAX_ENC, labels = paste("Encounter", 1:MAX_ENC), expand = expansion(add = 0.35)) +
  labs(x = NULL, y = "Mean Stop Duration (seconds)\n(Post Event - Pre Event)", colour = NULL,
       title = "Adaptation to multisensory events as an index of attentional control") +
  theme_eye() + theme(legend.position = "bottom", legend.text = element_text(size = 18), axis.text.x = element_text(size = 16))
save(f5, "fig5_ms_adaptation.png", 16, 7)
write.csv(s5[, c("Group", "Event", "encounter", "m", "se", "n")], file.path(root, "results/figure5_means.csv"), row.names = FALSE)

# ---------------------------------------------------------------- Figure 6: search success
lims_bt <- c(0, max(D$BT) * 1.03)
f6c <- ggplot(D, aes(BT, success_proportion, colour = Group, fill = Group)) +
  geom_smooth(method = "lm", formula = y ~ x, alpha = 0.18, linewidth = 1.3) +
  geom_point(size = 3.2, alpha = 0.75) +
  scale_colour_manual(values = c(Children = TEAL_L, Adults = ORANGE_D)) + scale_fill_manual(values = FILL) +
  scale_y_continuous(limits = c(0, 1.14), breaks = brk(c(0, 1.14)), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Total Backtracking Duration (seconds)", y = NULL, colour = NULL, fill = NULL) +
  theme_eye() + theme(legend.position = "inside", legend.position.inside = c(0.02, 0.02), legend.justification = c(0, 0), legend.text = element_text(size = 16))
f6 <- (bars("success_proportion", "Proportion of Targets Found", c(0, 1.14)) | age_scatter("success_proportion", c(0, 1.14)) | f6c) +
  plot_annotation(title = "Search Performance", theme = theme(plot.title = element_text(size = 26, face = "bold", hjust = 0.5)))
save(f6, "fig6_search.png", 19, 6.3)

# ---------------------------------------------------------------- Figure S1: regression scatterplots
# Each panel: one outcome against one predictor, with each age group's OLS line and 95% CI after the
# outlier rule of that model (as in the text); open circles mark participants the rule excluded.
panel <- function(y, x, ylab, xlab) {
  ex <- unlist(PS[[paste0("single_", y, "_", x)]]$excluded)
  X <- D[!is.na(D[[x]]) & !is.na(D[[y]]), ]; X$excluded <- X$code %in% ex
  ggplot(X, aes(.data[[x]], .data[[y]], colour = Group, fill = Group)) +
    geom_smooth(data = X[!X$excluded, ], method = "lm", formula = y ~ x, alpha = 0.15, linewidth = 1.2) +
    geom_point(data = X[!X$excluded, ], size = 2.6, alpha = 0.75) +
    geom_point(data = X[X$excluded, ], size = 2.8, shape = 21, fill = NA, stroke = 1.1, show.legend = FALSE) +
    scale_colour_manual(values = c(Children = TEAL_L, Adults = ORANGE_D)) + scale_fill_manual(values = FILL) +
    coord_cartesian(ylim = range(X[[y]]) + c(-0.03, 0.03) * diff(range(X[[y]]))) +
    labs(x = xlab, y = ylab, colour = NULL, fill = NULL) + theme_eye(16) + theme(legend.position = "none")
}
BTL <- "Backtracking Duration (seconds)"; ADL <- "Adaptation Slope\n(seconds per encounter)"
fs1 <- (panel("spatial_coverage", "BT", "Spatial Coverage", BTL) | panel("spatial_coverage", "adaptation", NULL, ADL)) /
       (panel("opt1", "BT", "Object Coverage", BTL) | panel("opt1", "adaptation", NULL, ADL)) /
       (panel("success_proportion", "BT", "Proportion of Targets Found", BTL) | panel("success_proportion", "adaptation", NULL, ADL)) +
  plot_layout(guides = "collect") + plot_annotation(tag_levels = "a") &
  theme(legend.position = "bottom", legend.text = element_text(size = 16), plot.tag = element_text(size = 22, face = "bold"))
save(fs1, "figS1_regressions.png", 11, 13)
