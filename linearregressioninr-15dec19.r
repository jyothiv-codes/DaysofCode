#15 dec 19-jyothi
data("longley") #using the longley dataset(in-built)
head(longley,5) #to print the first 5 rows in the dataset
?longley  #documentation that provides details of all the attributes; references and examples are also provided
print(nrow(longley))  #number of rows
print(ncol(longley))  #number of columns
linearreg <- lm(Employed ~ Population, data=longley)  #Employed is the dependent variable and Population is the independent var
print(linearreg)  #prints the co-efficients
summary(linearreg)  #provides the quartile values, residual standard error and other statistics
predict(linearreg, data.frame(Population = 116.453))  #for a given population value, predict the number of people employed 
