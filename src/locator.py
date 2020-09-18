from matplotlib import pyplot as plt


# Un locator qui affiche le premier du mois et s'il y a la place, le dernier jour.
# Hardcodé pour l'index des données en dates DD/MM/YYYY
class FirstOfMonthLocator(plt.Locator):
        
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        # Tous les premier du mois
        result = []
        for  d in range(int(vmin), int(vmax)):
            date = self.axis.major.formatter(d,0);
            day = date[:2]
            if day == "01":
                result.append(d)
        # add the last if larger than 20
        lastday = self.axis.major.formatter(int(vmax) - 1,0)[:2]
        try:
            if int(lastday) > 20:
                result.append(int(vmax) - 1);
        except ValueError:
            # Silently ignore the harmless error
            pass
            
        return result




