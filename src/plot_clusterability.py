from helpers.utils import read_xml, calculate_clusterability


def main():
  files = ['CMT01.xml','CMT02.xml','CMT03.xml','CMT04.xml','CMT05.xml','CMT06.xml','CMT07.xml','CMT08.xml',
                    'CMT09.xml','CMT10.xml','CMT11.xml','CMT12.xml','CMT13.xml','CMT14.xml',
                    'M-n101-k10.xml','M-n121-k07.xml','M-n151-k12.xml','M-n200-k16.xml','M-n200-k17.xml',
                    'Golden_01.xml','Golden_02.xml','Golden_03.xml','Golden_04.xml','Golden_05.xml','Golden_06.xml',
                    'Golden_07.xml','Golden_08.xml','Golden_09.xml','Golden_10.xml','Golden_11.xml','Golden_12.xml',
                    'Golden_13.xml','Golden_14.xml','Golden_15.xml','Golden_16.xml','Golden_17.xml','Golden_18.xml',
                    'Golden_19.xml','Golden_20.xml',
                    'Li_21.xml','Li_22.xml','Li_23.xml','Li_24.xml','Li_25.xml','Li_26.xml','Li_27.xml','Li_28.xml',
                    'Li_29.xml','Li_30.xml','Li_31.xml','Li_32.xml']
    
  X_files = ['X-n101-k25.xml','X-n106-k14.xml','X-n110-k13.xml','X-n115-k10.xml',
          'X-n120-k6.xml','X-n125-k30.xml','X-n129-k18.xml','X-n134-k13.xml',
          'X-n139-k10.xml','X-n143-k7.xml','X-n148-k46.xml','X-n153-k22.xml',
          'X-n157-k13.xml','X-n162-k11.xml','X-n167-k10.xml','X-n172-k51.xml',
          'X-n176-k26.xml','X-n181-k23.xml','X-n186-k15.xml','X-n190-k8.xml',
          'X-n195-k51.xml','X-n200-k36.xml','X-n204-k19.xml','X-n209-k16.xml',
          'X-n214-k11.xml','X-n219-k73.xml','X-n223-k34.xml','X-n228-k23.xml',
          'X-n233-k16.xml','X-n237-k14.xml','X-n242-k48.xml','X-n247-k47.xml',
          'X-n251-k28.xml','X-n256-k16.xml','X-n261-k13.xml','X-n266-k58.xml',
          'X-n270-k35.xml','X-n275-k28.xml','X-n280-k17.xml','X-n284-k15.xml',
          'X-n289-k60.xml','X-n294-k50.xml','X-n298-k31.xml','X-n303-k21.xml',
          'X-n308-k13.xml','X-n313-k71.xml','X-n317-k53.xml','X-n322-k28.xml',
          'X-n327-k20.xml','X-n331-k15.xml','X-n336-k84.xml','X-n344-k43.xml',
          'X-n351-k40.xml','X-n359-k29.xml','X-n367-k17.xml','X-n376-k94.xml',
          'X-n384-k52.xml','X-n393-k38.xml','X-n401-k29.xml','X-n411-k19.xml',
          'X-n420-k130.xml','X-n429-k61.xml','X-n439-k37.xml','X-n449-k29.xml',
          'X-n459-k26.xml','X-n469-k138.xml','X-n480-k70.xml','X-n491-k59.xml',
          'X-n502-k39.xml','X-n513-k21.xml','X-n524-k137.xml','X-n536-k96.xml',
          'X-n548-k50.xml','X-n561-k42.xml','X-n573-k30.xml','X-n586-k159.xml',
          'X-n599-k92.xml','X-n613-k62.xml','X-n627-k43.xml','X-n641-k35.xml',
          'X-n655-k131.xml','X-n670-k126.xml','X-n685-k75.xml','X-n701-k44.xml',
          'X-n716-k35.xml','X-n733-k159.xml','X-n749-k98.xml','X-n766-k71.xml',
          'X-n783-k48.xml','X-n801-k40.xml','X-n819-k171.xml','X-n837-k142.xml',
          'X-n856-k95.xml','X-n876-k59.xml','X-n895-k37.xml','X-n916-k207.xml',
          'X-n936-k151.xml','X-n957-k87.xml','X-n979-k58.xml','X-n1001-k43.xml']
    
  print("Problem,Dip score,Dip P Value")
  for problem in files + X_files:
      problem_name = problem.split(".xml")[0]

      if problem[0]=="X":
        save_file= "./results/clusterability/X_datasets/"+problem_name+"_clusterability.png"
      else:
        save_file= "./results/clusterability/"+problem_name+"_clusterability.png"

      _, _, _, customer_locations, _ = read_xml(
            './data/'+problem,0, False, False)

      dip, pval = calculate_clusterability(customer_locations, plot=True, save_file=save_file)
      print("{},{:.5f},{:.5f}".format(problem, dip, pval))


if __name__ == "__main__":
    main()