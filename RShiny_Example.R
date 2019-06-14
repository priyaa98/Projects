#User Interface with fluidity (i.e.) can adapt to any window
ui1 = fluidPage(
  titlePanel("Comparing survey dataset columns"),
  sidebarLayout(
    sidebarPanel (
    #inputs given to the server to provide plotted output
      radioButtons("p", "Select an attribute of dataset:",
                   list("Attribute 1"='a', "Attribute 2"='b', "Attribute 3"='c', "Attribute 4"='d',"Attribute 5"='e')),
      radioButtons("q", "Select another column of survey dataset:",
                   list("Attribute 1"='a', "Attribute 2"='b', "Attribute 3"='c', "Attribute 4"='d',"Attribute 5"='e'))
    ),
    #plotted output based on chosen inputs --> connection with the server/ functional part of the program
    mainPanel(plotOutput("distPlot"))
  )
)

#Server (or) the functional part of the webpage --> All functions and processing happen in this part of the program.
server1 = function(input, output) {
#processing for the plotted output
  output$distPlot <- renderPlot({
  #to plot based on the input given in the UI
  #input taken from the first set of radio buttons tag named "p"
    if(input$p=='a'){
      i<-1
    }
    if(input$p=='b'){
      i<-2
    }
    if(input$p=='c'){
      i<-3
    }
    if(input$p=='d'){
      i<-4
    }
    if(input$p=='e'){
      i<-5
    }
    
    #input taken from the second set of radio buttons tag named "q"
    if(input$q=='a'){
      j<-1
    }
    if(input$q=='b'){
      j<-2
    }
    if(input$q=='c'){
      j<-3
    }
    if(input$q=='d'){
      j<-4
    }
    if(input$q=='e'){
      j<-5
    }
    
    Plotting based on the given inputs
    x<- s[, i]
    y <- s[, j]
    plot(x,y,type = "p", xlab = cols_S[i], ylab = cols_S[j])
  })
}

#R Shiny's function shinyApp works best in the latest version of R (preferably >3.0.0)
shinyApp(ui1,server1)
