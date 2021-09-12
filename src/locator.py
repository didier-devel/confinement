from matplotlib import pyplot as plt


# Un locator qui affiche le premier des mois pair
# Hardcodé pour l'index des données en dates DD/MM/YYYY
class FirstOfMonthLocator(plt.Locator):
        
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        # Tous les premier du mois
        result = []
        for  d in range(int(vmin), int(vmax)):
            date = self.axis.major.formatter(d,0);
            day = date[:2]
            month = date[3:5]
            m = int(month)%2
            if day == "01" and m == 0:
                result.append(d)


            
        return result




