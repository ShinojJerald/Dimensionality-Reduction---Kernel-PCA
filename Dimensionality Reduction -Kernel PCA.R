
############ Kernel PCA ##################
stats = read.csv('Titanictest.csv')
sum(is.na(stats))
stats$Age = ifelse(is.na(stats$Age),
                   ave(stats$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                   stats$Age)


#Encoding/changing categorical values as factors

stats$Sex = as.numeric(factor(stats$Sex,
                              levels = c('male','female'),
                              labels = c(0 , 1)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(stats$Survived, SplitRatio = 0.8)
training_set = subset(stats, split == TRUE)
test_set = subset(stats, split == FALSE)


# Feature Scaling
training_set[, -5] = scale(training_set[, -5])
test_set[, -5] = scale(test_set[, -5])

# Applying Kernel PCA
# install.packages('kernlab')
library(kernlab)
kpca = kpca(~., data = training_set[-5], kernel = 'rbfdot', features = 2)
training_set_pca = as.data.frame(predict(kpca, training_set))
training_set_pca$Survived = training_set$Survived
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Survived = test_set$Survived

# Fitting Logistic Regression to the Training set
classifier = glm(formula = Survived ~ .,
                 family = binomial,
                 data = training_set_pca)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set_pca[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred
# Making the Confusion Matrix
cm = table(test_set_pca[, 3], y_pred)
cm
Accuracy=(cm[1]+cm[4])/(cm[1]+cm[4]+cm[3]+cm[2])
Accuracy
#Accuracy for Kernalized PCA = 77.5%

# Visualising the Training set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
# install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set_pca
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('V1', 'V2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


