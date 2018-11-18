f = open("Submission.csv","r")
f_out = open("Submission_out.csv","w")
for line in f.readlines():
  line = line.split(',')
  line[0] = str(int(line[0]) + 1)
  line = ",".join(line)
  f_out.write(line)
f.close()
f_out.close()
