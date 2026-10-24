# Event-level mixed models.
#  1-3. Adaptation to multisensory (MS) events: change in stopping time (post - pre) by age group,
#       event type (MS vs control) and encounter (first four encounters, counted across blocks).
#       Degrees of freedom: Kenward-Roger; post-hoc contrasts Tukey-adjusted within each family.
#  4.   Supplement: trial-level search model (logistic, crossed random effects).
#  5.   Supplement: stopping measured as the longest pause in each window.
# Run from the repository root or from analysis/. Outputs: results/event_models.{txt,json}
suppressMessages({library(lmerTest); library(emmeans); library(jsonlite)})
root <- if (dir.exists("data")) "." else ".."
E <- read.csv(file.path(root, "data/events.csv"))
P <- read.csv(file.path(root, "data/participants.csv"))
S <- read.csv(file.path(root, "data/subject_measures.csv"))
E$group   <- factor(E$group, levels = c("adult", "child"))
E$MSevent <- factor(E$event_type == "MS", levels = c(FALSE, TRUE))
MAX_ENC <- 4
E4 <- E[E$encounter <= MAX_ENC, ]; E4$encF <- factor(E4$encounter)
emm_options(lmer.df = "kenward-roger", lmerTest.limit = 1e5, pbkrtest.limit = 1e5)
out <- list(); sink(file.path(root, "results/event_models.txt"), split = TRUE)
co <- function(m, term) {
  s <- summary(m, ddf = "Kenward-Roger")$coefficients[term, ]; ci <- confint(m, parm = term, method = "Wald")
  list(b = s[["Estimate"]], CI = as.numeric(ci), t = s[["t value"]], df = s[["df"]], p = s[["Pr(>|t|)"]])
}
cat("events (encounters 1-4):", nrow(E4), "| MS:", sum(E4$MSevent == "TRUE"), "| participants:", length(unique(E4$code)), "\n")
out$n_events <- nrow(E4)

# 1. three-way interaction, encounter numeric
m1 <- lmer(stop_diff ~ group * MSevent * encounter + (1 | code), data = E4)
out$threeway <- co(m1, "groupchild:MSeventTRUE:encounter")
me <- as.data.frame(pairs(emmeans(m1, ~ group)))
out$group_main_effect <- list(estimate = -me$estimate, t = -me$t.ratio, df = me$df, p = me$p.value)
cat(sprintf("3-way group x MS x encounter: b=%.3f [%.3f, %.3f], t(%.0f)=%.2f, p=%.4f\n", out$threeway$b, out$threeway$CI[1], out$threeway$CI[2], out$threeway$df, out$threeway$t, out$threeway$p))
cat(sprintf("group main effect, averaged over event type and encounters (children - adults): %.3f, t(%.0f)=%.2f, p=%.3f\n", out$group_main_effect$estimate, out$group_main_effect$df, out$group_main_effect$t, out$group_main_effect$p))
tr <- as.data.frame(summary(emtrends(m1, ~ group * MSevent, var = "encounter"), infer = TRUE))
cat("\nSlope of encounter by group and event type:\n"); print(tr, digits = 3, row.names = FALSE)
out$encounter_slopes <- tr

# 2. encounter as factor: omnibus test and post-hoc contrasts
m3 <- lmer(stop_diff ~ group * MSevent * encF + (1 | code), data = E4)
a3 <- anova(m3, ddf = "Kenward-Roger")["group:MSevent:encF", ]
out$omnibus <- list(F = a3[["F value"]], df1 = a3[["NumDF"]], df2 = a3[["DenDF"]], p = a3[["Pr(>F)"]])
cat(sprintf("\nomnibus 3-way (encounter as factor): F(%d, %.0f)=%.2f, p=%.4f\n", a3[["NumDF"]], a3[["DenDF"]], a3[["F value"]], a3[["Pr(>F)"]]))
pe <- as.data.frame(summary(pairs(emmeans(m3, ~ encF | group * MSevent), adjust = "tukey"), infer = TRUE))
cat("\nEncounter contrasts (Tukey within group x event type):\n"); print(pe[, c("contrast", "group", "MSevent", "estimate", "lower.CL", "upper.CL", "df", "t.ratio", "p.value")], digits = 3, row.names = FALSE)
out$encounter_contrasts <- pe
mc <- as.data.frame(summary(pairs(emmeans(m3, ~ MSevent | group * encF)), infer = TRUE))
cat("\nMS vs control at each encounter (estimate = control - MS):\n"); print(mc[, c("group", "encF", "estimate", "lower.CL", "upper.CL", "df", "t.ratio", "p.value")], digits = 3, row.names = FALSE)
out$ms_vs_control <- mc
e1 <- as.data.frame(summary(pairs(emmeans(m3, ~ group | MSevent * encF)), infer = TRUE))
e1 <- e1[e1$MSevent == "TRUE" & e1$encF == "1", ]
out$adult_vs_child_enc1 <- list(estimate = e1$estimate, CI = c(e1$lower.CL, e1$upper.CL), t = e1$t.ratio, df = e1$df, p = e1$p.value)
cat(sprintf("\nAdults - children, MS encounter 1: %.3f s [%.3f, %.3f], t(%.0f)=%.2f, p=%.3f\n", e1$estimate, e1$lower.CL, e1$upper.CL, e1$df, e1$t.ratio, e1$p.value))

# 3. children only: age (centered) x event type x encounter
K <- E4[E4$group == "child", ]
K$age <- P$age[match(K$code, P$code)]
K$age_c <- K$age - mean(unique(K[, c("code", "age")])$age)
mk <- lmer(stop_diff ~ age_c * MSevent * encF + (1 | code), data = K)
ak <- anova(mk, ddf = "Kenward-Roger")
msk <- as.data.frame(summary(pairs(emmeans(mk, ~ MSevent)), infer = TRUE))
out$children_ms_effect <- list(estimate = -msk$estimate, CI = -rev(c(msk$lower.CL, msk$upper.CL)), t = -msk$t.ratio, df = msk$df, p = msk$p.value)
out$children_age_x_ms <- list(F = ak["age_c:MSevent", "F value"], df2 = ak["age_c:MSevent", "DenDF"], p = ak["age_c:MSevent", "Pr(>F)"])
cat(sprintf("\nChildren only (n events=%d): MS - control (mean age, averaged over encounters)=%.3f [%.3f, %.3f], t(%.0f)=%.2f, p=%.4f; age x MS: F(1, %.0f)=%.2f, p=%.3f\n",
            nrow(K), out$children_ms_effect$estimate, out$children_ms_effect$CI[1], out$children_ms_effect$CI[2], msk$df, out$children_ms_effect$t, msk$p.value,
            out$children_age_x_ms$df2, out$children_age_x_ms$F, out$children_age_x_ms$p))

# 4. supplement: trial-level search model (all trials, no exclusions)
Tr <- read.csv(file.path(root, "data/search_trials.csv"))
Tr$BT <- S$BT[match(Tr$code, S$code)]; Tr$group <- factor(S$group[match(Tr$code, S$code)], levels = c("adult", "child"))
bt <- unique(Tr[, c("code", "BT")]); Tr$BTz <- (Tr$BT - mean(bt$BT)) / sd(bt$BT)
g <- glmer(success ~ BTz * group + (1 | code) + (1 | target), data = Tr, family = binomial, control = glmerControl(optimizer = "bobyqa"))
gs <- summary(g)$coefficients
sl <- as.data.frame(summary(emtrends(g, ~ group, var = "BTz"), infer = TRUE))
out$search_glmm <- list(n_trials = nrow(Tr), n_participants = length(unique(Tr$code)),
                        interaction = list(b = gs["BTz:groupchild", "Estimate"], se = gs["BTz:groupchild", "Std. Error"], CI = gs["BTz:groupchild", "Estimate"] + c(-1, 1) * qnorm(.975) * gs["BTz:groupchild", "Std. Error"], z = gs["BTz:groupchild", "z value"], p = gs["BTz:groupchild", "Pr(>|z|)"]),
                        slopes = sl)
cat(sprintf("\nSupplement: trial-level search model (%d trials, %d participants): backtracking x group b=%.3f, z=%.2f, p=%.3f\n",
            nrow(Tr), length(unique(Tr$code)), gs["BTz:groupchild", "Estimate"], gs["BTz:groupchild", "z value"], gs["BTz:groupchild", "Pr(>|z|)"]))
print(sl, digits = 3, row.names = FALSE)

# 5. supplement: stopping measured as the longest pause in each 5-s window, rebuilt from the teleport times.
#    The main measure is the mean interval between teleports (one teleport: the longer time to a window
#    edge; none: 5 s), which cannot tell even from uneven pauses. The main model is repeated with the
#    longest teleport-free interval in the window (window edges included).
TP <- read.csv(file.path(root, "data/teleports.csv")); tpl <- split(TP$time, paste(TP$code, TP$block))
eps <- 1e-6                                    # event and teleport times are logged to the millisecond
measures <- function(tt, a, b) {               # teleports in [a, b)
  x <- sort(tt[tt > a - eps & tt < b - eps])
  m <- if (length(x) > 1) mean(diff(x)) else if (length(x) == 1) max(x - a, b - x) else b - a
  c(mean_interval = m, longest = max(diff(c(a, x, b))), n = length(x))
}
A <- t(mapply(function(code, block, ws) {
  tt <- tpl[[paste(code, block)]]; e <- ws + 5
  measures(tt, e, e + 5) - measures(tt, e - 5, e)
}, E$code, E$block, E$window_start))
E$alt_mean <- A[, "mean_interval"]; E$alt_longest <- A[, "longest"]
cat(sprintf("\nSupplement: stopping measure rebuilt from teleport times; max |difference| from the main measure = %.1e\n", max(abs(E$alt_mean - E$stop_diff))))
E4 <- E[E$encounter <= MAX_ENC, ]; E4$encF <- factor(E4$encounter); alt <- list()
for (v in c("alt_longest")) {
  m1v <- lmer(as.formula(paste(v, "~ group * MSevent * encounter + (1 | code)")), data = E4)
  m3v <- lmer(as.formula(paste(v, "~ group * MSevent * encF + (1 | code)")), data = E4)
  s <- co(m1v, "groupchild:MSeventTRUE:encounter")
  mcv <- as.data.frame(summary(pairs(emmeans(m3v, ~ MSevent | group * encF)), infer = TRUE))
  trv <- as.data.frame(summary(emtrends(m1v, ~ group * MSevent, var = "encounter"), infer = TRUE))
  alt[[v]] <- list(threeway = s, ms_vs_control = mcv, slopes = trv)
  print(trv, digits = 3, row.names = FALSE)
  cat(sprintf("%s: 3-way group x MS x encounter b=%.3f, t(%.0f)=%.2f, p=%.4f; MS vs control (estimate = control - MS):\n",
              "longest pause (s)", s$b, s$df, s$t, s$p))
  print(mcv[, c("group", "encF", "estimate", "df", "t.ratio", "p.value")], digits = 3, row.names = FALSE)
}
out$stopping_measures <- alt
sink(); write_json(out, file.path(root, "results/event_models.json"), auto_unbox = TRUE, digits = 8)
