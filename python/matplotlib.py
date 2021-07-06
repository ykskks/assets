# y軸の数字の表記を見やすくする
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: ":,".format(int(x))))

# xticks回転
ax.tick_params(axis="x", rotation=90)
