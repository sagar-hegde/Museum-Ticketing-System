import json
import os
import uuid
from datetime import datetime
import qrcode

DATA_FILE = "tickets.json"

MUSEUMS = {
    1: "National History Museum",
    2: "Art Gallery Museum",
    3: "Science Museum",
    4: "Heritage Museum"
}

def load_tickets():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_tickets(tickets):
    with open(DATA_FILE, "w") as f:
        json.dump(tickets, f, indent=4)

def login():
    print("\n===== MeseoTicket Login =====")
    username = input("Enter Username: ")
    password = input("Enter Password: ")
    print("Login Successful!\n")
    return username

def select_museum():
    print("Available Museums:")
    for key, value in MUSEUMS.items():
        print(f"{key}. {value}")
    choice = int(input("Select Museum: "))
    return MUSEUMS.get(choice, "Unknown Museum")

def generate_qr(ticket_id):
    img = qrcode.make(ticket_id)
    filename = f"{ticket_id}.png"
    img.save(filename)
    return filename

def book_ticket(user):
    tickets = load_tickets()

    museum = select_museum()
    visitors = int(input("Number of Visitors: "))
    visit_date = input("Visit Date (YYYY-MM-DD): ")

    ticket_id = str(uuid.uuid4())[:8]

    qr_file = generate_qr(ticket_id)

    ticket = {
        "Ticket ID": ticket_id,
        "User": user,
        "Museum": museum,
        "Visitors": visitors,
        "Visit Date": visit_date,
        "Booked On": str(datetime.now()),
        "QR Code": qr_file
    }

    tickets.append(ticket)
    save_tickets(tickets)

    print("\n===== Ticket Booked Successfully =====")
    print("Ticket ID :", ticket_id)
    print("Museum    :", museum)
    print("Visitors  :", visitors)
    print("Visit Date:", visit_date)
    print("QR Saved  :", qr_file)

def view_tickets(user):
    tickets = load_tickets()

    print("\n===== My Tickets =====")

    found = False

    for ticket in tickets:
        if ticket["User"] == user:
            found = True
            print("-" * 40)
            print("Ticket ID :", ticket["Ticket ID"])
            print("Museum    :", ticket["Museum"])
            print("Visitors  :", ticket["Visitors"])
            print("Visit Date:", ticket["Visit Date"])
            print("QR Code   :", ticket["QR Code"])

    if not found:
        print("No Tickets Found.")

def main():
    user = login()

    while True:
        print("\n====== MeseoTicket ======")
        print("1. Book Ticket")
        print("2. View My Tickets")
        print("3. Exit")

        choice = input("Enter Choice: ")

        if choice == "1":
            book_ticket(user)

        elif choice == "2":
            view_tickets(user)

        elif choice == "3":
            print("\nThank you for using MeseoTicket!")
            print("Paperless • Contactless • Eco-Friendly")
            break

        else:
            print("Invalid Choice!")

def loadData(dataname):
    filename = home_dir + 'Resource/' + dataname + '.npz'
    npzfile = numpy.load(filename)
    print(npzfile.files)
    X = npzfile['X']
    y = npzfile['y']
    n, d = X.shape
    #X = numpy.concatenate((X, numpy.ones((n, 1))), axis=1)
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    return X, y

def experiment(xMat, yVec, maxiter, repeat, gamma, isSearch, isExact, newtoniter=100):
    demo = Demo(maxiter, repeat, gamma)
    demo.fit(xMat, yVec, m=256)
    condnum = demo.condnum
    print('Condition number is ' + str(condnum))
    
    m = 4
    print('m = ' + str(m))
    err1 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 16
    print('m = ' + str(m))
    err2 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 64
    print('m = ' + str(m))
    err3 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 256
    print('m = ' + str(m))
    err4 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    
    return err1, err2, err3, err4, condnum


def plotConvergence(outfilename, imagename):
    npzfile = numpy.load(outfilename)
    dataname = str(npzfile['dataname'])
    maxiter = npzfile['maxiter']
    newtoniter = npzfile['newtoniter']
    err1 = npzfile['err1']
    err2 = npzfile['err2']
    err3 = npzfile['err3']
    err4 = npzfile['err4']
    
    repeat = err1.shape[0]
    
    # plot
    fig = plt.figure(figsize=(9, 8))
    
    for r in range(repeat):
        plt.semilogy(err1[r, :], color='greenyellow', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err2[r, :], color='salmon', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err3[r, :], color='skyblue', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err4[r, :], color='grey', linestyle='-', linewidth=0.5, alpha=0.5)  
    
    line0, = plt.semilogy(numpy.median(err1, axis=0), color='g', linestyle='-', linewidth=3)
    line1, = plt.semilogy(numpy.median(err2, axis=0), color='r', linestyle='-', linewidth=3.5)
    line2, = plt.semilogy(numpy.median(err3, axis=0), color='b', linestyle='-', linewidth=3.5)
    line3, = plt.semilogy(numpy.median(err4, axis=0), color='k', linestyle='-', linewidth=3.5)
    
    fontsize = 32
    #plt.legend([line0, line1, line2, line3], ['m=4', 'm=16', 'm=64', 'm=256'], fontsize=20)
    plt.xlabel('iterations (t)', fontsize=fontsize+2)
    #plt.ylabel(r"$|| w - w^\star ||_2$", fontsize=fontsize)
    #plt.title(r"$\gamma = 10^{-2}$", fontsize=fontsize+2)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=fontsize) 
    plt.yticks(fontsize=fontsize) 
    plt.axis([0, maxiter, 1e-8, 5])
    plt.tight_layout()
    
    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)
    #plt.show()
    
def main(NewtonIter, Gamma, ResultName): 
    #dataname = 'logis_U8'
    dataname = 'covtype'
    path = home_dir + 'Output/logis/'
    MaxIter = 30
    Repeat = 10
    IsSearch = False
    IsExact = False
    
    xMat, yVec = loadData(dataname)
    print(xMat)
    print(yVec)
    
    err1, err2, err3, err4, condnum = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact, newtoniter=NewtonIter)
    outfilename = path + dataname + ResultName + '.npz'
    numpy.savez(outfilename, err1=err1, err2=err2, err3=err3, err4=err4, dataname=dataname, maxiter=MaxIter, newtoniter=NewtonIter, condnum=condnum)
    
    imagename = path + dataname + ResultName + '.pdf'
    plotConvergence(outfilename, imagename)
    
def loadData(dataname):
    filename = home_dir + 'Resource/' + dataname + '.npz'
    npzfile = numpy.load(filename)
    print(npzfile.files)
    X = npzfile['X']
    y = npzfile['y']
    n, d = X.shape
    #X = numpy.concatenate((X, numpy.ones((n, 1))), axis=1)
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    return X, y

def experiment(xMat, yVec, maxiter, repeat, gamma, isSearch, isExact, newtoniter=100):
    demo = Demo(maxiter, repeat, gamma)
    demo.fit(xMat, yVec, m=256)
    condnum = demo.condnum
    print('Condition number is ' + str(condnum))
    
    m = 4
    print('m = ' + str(m))
    err1 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 16
    print('m = ' + str(m))
    err2 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 64
    print('m = ' + str(m))
    err3 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 256
    print('m = ' + str(m))
    err4 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    
    return err1, err2, err3, err4, condnum


def plotConvergence(outfilename, imagename):
    npzfile = numpy.load(outfilename)
    dataname = str(npzfile['dataname'])
    maxiter = npzfile['maxiter']
    err1 = npzfile['err1']
    err2 = npzfile['err2']
    err3 = npzfile['err3']
    err4 = npzfile['err4']
    
    repeat = err1.shape[0]
    
    condnum = npzfile['condnum']
    print('Condition number is ' + str(condnum))
    
    # plot
    fig = plt.figure(figsize=(9, 8))
    
    for r in range(repeat):
        plt.semilogy(err1[r, :], color='greenyellow', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err2[r, :], color='salmon', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err3[r, :], color='skyblue', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err4[r, :], color='grey', linestyle='-', linewidth=0.5, alpha=0.5)  
    
    line0, = plt.semilogy(numpy.median(err1, axis=0), color='g', linestyle='-', linewidth=3)
    line1, = plt.semilogy(numpy.median(err2, axis=0), color='r', linestyle='-', linewidth=3.5)
    line2, = plt.semilogy(numpy.median(err3, axis=0), color='b', linestyle='-', linewidth=3.5)
    line3, = plt.semilogy(numpy.median(err4, axis=0), color='k', linestyle='-', linewidth=3.5)
    
    fontsize = 32
    #plt.legend([line0, line1, line2, line3], ['m=4', 'm=16', 'm=64', 'm=256'], fontsize=20)
    plt.xlabel('iterations (t)', fontsize=fontsize+2)
    #plt.ylabel(r"$|| w - w^\star ||_2$", fontsize=fontsize)
    #plt.title(r"$\gamma = 10^{-2}$", fontsize=fontsize+2)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=fontsize) 
    plt.yticks(fontsize=fontsize) 
    plt.axis([0, 20, 1e-12, 5])
    plt.tight_layout()
    
    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)
    #plt.show()
    
def main(NewtonIter, Gamma, ResultName): 
    dataname = 'logis_N8'
    path = home_dir + 'Output/logis/'
    MaxIter = max(20, int(numpy.ceil(1000 / (NewtonIter+10))))
    Repeat = 10
    IsSearch = True
    IsExact = False
    
    xMat, yVec = loadData(dataname)
    print(xMat)
    print(yVec)
    
    err1, err2, err3, err4, condnum = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact, newtoniter=NewtonIter)
    outfilename = path + dataname + ResultName + '.npz'
    numpy.savez(outfilename, err1=err1, err2=err2, err3=err3, err4=err4, dataname=dataname, maxiter=MaxIter, newtoniter=NewtonIter, condnum=condnum)
    
    imagename = path + dataname + ResultName + '.pdf'
    plotConvergence(outfilename, imagename)
    
    

if __name__ == '__main__': 
    Gamma = 1e-6
    GammaName = '_gamma-6'
    
    NewtonIter = 270
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

    Gamma = 1e-8
    GammaName = '_gamma-8'

    NewtonIter = 270
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    
    
    
    Gamma = 1e-10
    GammaName = '_gamma-10'

    NewtonIter = 270
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)    

if __name__ == '__main__':
    
    NewtonIter = 30
    Gamma = 1e-2
    GammaName = '_gamma-2'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    NewtonIter = 90
    Gamma = 1e-3
    GammaName = '_gamma-3'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    NewtonIter = 270
    Gamma = 1e-4
    GammaName = '_gamma-4'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    NewtonIter = 810
    Gamma = 1e-5
    GammaName = '_gamma-5'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)

if __name__ == "__main__":
    main()
