import matplotlib.pyplot as plt


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2], [3, 3.5]], [0, 1, 2, 3])
print reg.coef_

training = [0, 1, 2, 3.5]
fit = [0, 1, 2, 3]
grades_range = [0, 1, 2, 3]
fig=plt.figure()

ax=fig.add_axes([0,0,1,1])
ax.scatter(grades_range, training, color='r')
ax.scatter(grades_range, fit, color='b')

plt.plot(training, training, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

