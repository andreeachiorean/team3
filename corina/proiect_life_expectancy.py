import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


#Project goal: What factors influence life expectancy in different countries?

#1. Loading the dataset
df =pd.read_csv("Life Expectancy Data.csv")

print(df.head())
print("Dimensiunile setului de date:")
print(f"Numar de randuri_ {df.shape[0]}")
print(f"Numar de coloane: {df.shape[1]}")

print("Numele coloanelor:")
print(df.columns.tolist())

print(df.info)

#Valorile lipsa
print("Valori lipsa")
missing_values = df.isnull().sum()
missing_percentage = (missing_values /len(df)) *100
missing_df = pd.DataFrame({
    "missing_values": missing_values,
    "percentage" : missing_percentage.round(2)
})

print(missing_df[missing_df["missing_values"] > 0 ])

#Eliminam randurile cu valori lipsa la variabila "Life Expectancy" - variabila care e importanta pt noi
df_clean = df.dropna(subset=['Life expectancy ']).copy()
print(df_clean.head())

# Pentru alte coloane , le vom completa cu media

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())

print(f"Dupa curatare , avem {df_clean.shape[0]} randuri")
print(f"Valori lipsa ramase : {df_clean.isnull().sum().sum()}")

#Distributia sperantei de viata
plt.figure(figsize = (14,10))

plt.subplot(2,2,1)
sns.histplot(df_clean['Life expectancy '], kde = True, bins =30, color ="blue")
plt.title("Distributia Sperantei de viata", fontsize = 14, fontweight = 'bold')
plt.xlabel("Speranta de viata(ani)")
plt.ylabel("Frecventa")   # Frecventa - nr de tari/Observatii
plt.axvline(df_clean['Life expectancy '].mean(),color ="red", linestyle = "--",
            label = f"Media: {df_clean['Life expectancy '].mean(): .1f} an i")
plt.legend()
plt.show()

#Grafic 2: Speranta de viata pe ani
plt.subplot(2,2,2)
life_by_year = df_clean.groupby ("Year")['Life expectancy '].mean()
plt.plot(life_by_year.index, life_by_year.values, marker = 'o',color = "green",
         linewidth=2)
plt.title("Distributia Sperantei de viata(2000-2015", fontsize = 14, fontweight = 'bold')
plt.xlabel("An")
plt.ylabel("Speranta de viata medie(ani)")
plt.grid(True, alpha=0.3)

# Grafic 3:Top 10 tari cu speranta de viata cea mai mare
plt.subplot(2,2,3)
top_countries =df_clean.groupby('Country')['Life expectancy '].mean().nlargest(10)
sns.barplot(x=top_countries.values, y= top_countries.index,hue=top_countries.index, palette = "viridis", legend= False)
plt.title("Top 10 Tari - Speranta de viata cea mai mare", fontsize = 10, fontweight = 'bold')
plt.xlabel("Speranta de viata medie(ani)")



#Grafic 4. Top 10 tari cu sparanta de viata cea mai mica
plt.subplot(2,2,4)
bottom_countries = df_clean.groupby('Country')['Life expectancy '].mean().nsmallest(10)
sns.barplot(x=bottom_countries.values, y= bottom_countries.index, hue=bottom_countries.index,palette = "magma", legend = False)
plt.title("Top 10 tari - Speranta de viata cea mai mica", fontsize = 14, fontweight = 'bold')
plt.xlabel("Speranta de viata medie(ani)")

plt.tight_layout()
plt.show()

#Analiza corelatiilor

#Selectam doar coloanele numerice pentru analiza corelatiei
numeric_df = df_clean.select_dtypes(include=[np.number])

# Calculam matricea de corelatie
correlation_matrix = numeric_df.corr()

#Graficul matricei de corelatie
plt.figure(figsize=(16,12))
sns.heatmap(correlation_matrix,annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square = True, linewidths =0.5, cbar_kws={"shrink": .8})
plt.title("Matricea de corelatie intre Variabile", fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.show()

#Corelatia cu speranta de viata
print("Corelatia variabilelor cu speranta de viata:")
life_corr =correlation_matrix['Life expectancy '].sort_values(ascending=False)
for i, (var, corr) in enumerate(life_corr.items()):
    if var != 'Life expectancy ':
        print(f"{var:30}: {corr:+.3f}")


#Pregatirea datelor pentru model

#Selectam variabilele cu cea mai mare corelatie cu speranta de viata
#(alegem primele 5, excluzand speranta de viata insasi)

top_features = life_corr[1:6].index.tolist()  #primele 5 valori
print(f"Variabilele selectate pentru model:")
for i, feature in enumerate(top_features, 1):
    print(f"{i}. {feature}")


#Cream DataFrame-ul pt model
X = df_clean[top_features]  # Variabilele independente(predictori)
y = df_clean['Life expectancy ']   #Variabila dependenta (ce vrem sa prezicem)


#Impartim datele in set de antrenare (80%) si set de testare (20%)
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size =0.2, random_state = 42)

print(f"Dimensiuni seturi de date:")
print(f"Set antrenare : {X_train.shape[0]} randuri")
print(f"Set testare: {X_test.shape[0]} randuri")

# Construirea modelului de regresie

#Cream si antrenam modelul
model =LinearRegression()
model.fit(X_train, y_train)

#Coeficientii modelului

print("Coeficientii modelului de regresie:")
print(f"Intercept (b0): {model.intercept_:.3f}")
print("\nCoeficienti (b1,b2,...:")
for i, (feature, coef) in enumerate(zip(top_features, model.coef_),1):
    print(f"{i:2}.{feature:30}: {coef:+.3f}")


#Interpretare coeficienti:
print("Interpretare coeficienti:")
print("Pentru fiecare coeficient pozitiv , speranta de viata creste cu valoarea coeficientului")
print("Pentru fiecare coeficient negativ, speranta de viata scade cu valoarea coeficientului")
print("\nExemplu de interpretare:")
for feature , coef in zip(top_features, model.coef_):
    direction ="crestere" if coef > 0 else "scade"
    print(f"- O crestere cu 1 unitate in '{feature}' face ca speranta de viata sa {direction} cu {abs(coef):.3f} ani")



#Evaluarea Modelului

#Predictii pe setul de test
y_pred = model.predict(X_test)


#Calculam metricile de evaluare
mse= mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Rezultatele modelului:")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R2 Score (R2): {r2:.3f}")

print(f"\n Interpretare R2:")
print(f"Modelul nostru explica {r2*100:.1f}% din variatia sperantei de viata")
print("R2 variaza de la 0 la1, unde 1 inseamna o predictie perfecta")

#Graficul valorilor reale vs prezise
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha =0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw =2, label ="Predictia Perfecta")
plt.xlabel("Valori Reale(ani)")
plt.ylabel("Valori prezise (ani)")
plt.title("Valori reale vs Valori Prezise - Model Regresie", fontsize =14, fontweight ='bold')
plt.legend()
plt.grid(True, alpha = 0.3)


#Adaugam linii pentru RMSE
plt.axhline(y_test.mean() + rmse, color ='orange', linestyle =':', alpha=0.7,
            label= f"+-RMSE ({rmse:.1f}ani)")
plt.axhline(y_test.mean() - rmse, color ='orange', linestyle =':', alpha=0.7)

plt.legend()
plt.tight_layout()
plt.show()



#Folosirea modelului pentru predictii

#Facem cateva predictii pentru tari specifice
print("Predictii pentru tari specifice:")
print("(Folosind date medii pe perioada 2000-2015)\n")

country_avg =df_clean.groupby("Country")[top_features].mean()


#Alegem cateva tari

countries_to_predict = ['Germany', 'Brasil', 'India', 'Nigeria', 'Japan']

for country in countries_to_predict:
    if country in country_avg.index:
        #Pregatim datele pt tara
        country_data = country_avg.loc[country].values.reshape(1,-1)

        #Facem predictia
        prediction = model.predict(country_data)[0]

        #Valoarea reala medie
        actual_value = df_clean[df_clean['Country']== country]['Life expectancy '].mean()

        print(f"Un {country}:")
        print(f"   Speranta de viata reala: {actual_value:.1f} ani")
        print(f"   Speranta de viata prezisa: {prediction:.1f} ani")
        print(f"   Diferenta:{abs(actual_value-prediction):.1f} ani")
        print()


#Analiza rezidualelor(erorilor)

#Calculam reziduurile(diferenta intre valoarea reala si cea prezisa)

residuals =y_test - y_pred

#Graficul rezidualelor
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
sns.histplot(residuals, kde = True, bins = 30, color = 'purple')
plt.title("Distributia Rezidualelor", fontsize =14, fontweight ='bold' )
plt.xlabel("Eroare (ani)")
plt.ylabel("Frecventa")
plt.axvline(0,color ="red", linestyle ='--', linewidth = 2)


plt.subplot(1,2,2)
plt.scatter(y_pred, residuals, alpha = 0.5, color ="orange")
plt.axhline(y=0, color ="red", linestyle ='--', linewidth = 2)
plt.xlabel("Valori prezise(ani)")
plt.ylabel("Reziduale(ani)")
plt.title("Reziduale vs valori prezise", fontsize =14, fontweight ="bold")
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()


print("Analiza rezidualelor:")
print(f" Media rezidualelor : {residuals.mean():.3f}(ar trebui sa fie aproape de 0")
print(f"Deviatia standard : {residuals.std():.3f}")
print(f"Reziduale intre -5 si 5 ani : {((residuals.abs()<= 5).sum()/len(residuals)*100):1f}%")
print("\n O distributie normala a rezidualelor indica un model bun")
print("Rezidualele ar trebui sa fie distribuite aleator, fara tipar")