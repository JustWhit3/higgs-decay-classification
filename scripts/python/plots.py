#====================================================
#     MODULES
#====================================================
import pandas as pd 
import ROOT
import matplotlib.pyplot as plt
import numpy as np

def main():
    #====================================================
    #     DATA PREPARATION
    #====================================================
    model_outputs = pd.read_csv('model_outputs.csv')
    model_outputs['Label'] = pd.read_csv('dataset_higgs_challenge.csv')['Label']
    model_outputs['KaggleWeight'] = pd.read_csv('dataset_higgs_challenge.csv')['KaggleWeight']
    model_outputs['KaggleSet'] = pd.read_csv('dataset_higgs_challenge.csv')['KaggleSet']

    predictions_train = model_outputs['Predictions'][model_outputs['KaggleSet'] == 't']
    predictions_test = model_outputs['Predictions'][model_outputs['KaggleSet'] == 'v']
    weights_train = model_outputs['KaggleWeight'][model_outputs['KaggleSet'] == 't']
    weights_test = model_outputs['KaggleWeight'][model_outputs['KaggleSet'] == 'v']                     
    labels_train = model_outputs['Label'][model_outputs['KaggleSet'] == 't']
    labels_test = model_outputs['Label'][model_outputs['KaggleSet'] == 'v']

    predictions_train = (predictions_train - min(predictions_train)) / (max(predictions_train) - min(predictions_train))
    predictions_test = (predictions_test - min(predictions_test)) / (max(predictions_test) - min(predictions_test))

    train_signal = predictions_train[model_outputs['KaggleSet'] == 't'][model_outputs['Label']=='s']
    train_bkg = predictions_train[model_outputs['KaggleSet'] == 't'][model_outputs['Label']=='b']
    test_signal = predictions_test[model_outputs['KaggleSet'] == 'v'][model_outputs['Label']=='s']
    test_bkg = predictions_test[model_outputs['KaggleSet'] == 'v'][model_outputs['Label']=='b']

    weights_train_signal = model_outputs['KaggleWeight'][model_outputs['KaggleSet'] == 't'][model_outputs['Label']=='s']
    weights_train_bkg = model_outputs['KaggleWeight'][model_outputs['KaggleSet'] == 't'][model_outputs['Label']=='b']
    weights_test_signal = model_outputs['KaggleWeight'][model_outputs['KaggleSet'] == 'v'][model_outputs['Label']=='s']
    weights_test_bkg = model_outputs['KaggleWeight'][model_outputs['KaggleSet'] == 'v'][model_outputs['Label']=='b']


    #====================================================
    #     STYLE SETTINGS
    #====================================================
    ROOT.gROOT.SetStyle("ATLAS")

    c = ROOT.TCanvas("c", "", 750, 700)

    bins = 20
    hist_train_s = ROOT.TH1D("hist_train_s", "train signal", bins, 0, 1)
    hist_test_s = ROOT.TH1D("hist_test_s", "test signal", bins, 0, 1)
    hist_train_b = ROOT.TH1D("hist_train_b", "train bkg", bins, 0, 1)
    hist_test_b = ROOT.TH1D("hist_test_b", "test bkg", bins, 0, 1)


    #====================================================
    #     FIRST UNWEIGHTED AND NORMALIZED TO UNITY
    #====================================================
    for i in range(len(train_signal)):
        hist_train_s.Fill(train_signal.values[i])
    for i in range(len(test_signal)):
        hist_test_s.Fill(test_signal.values[i])
    for i in range(len(train_bkg)):
        hist_train_b.Fill(train_bkg.values[i])
    for i in range(len(test_bkg)):
        hist_test_b.Fill(test_bkg.values[i])


    for hist in [hist_test_s, hist_test_b]:
        for i in range(1, hist.GetNbinsX()+1):
            hist.SetBinError(i, np.sqrt(hist.GetBinContent(i)))
    for hist in [hist_train_s, hist_test_s, hist_train_b, hist_test_b]:
        hist.Scale(1/hist.Integral(), 'nosw2')

    #Plot settings:
    hist_train_b.SetAxisRange(3e-3, 5, 'Y')
    hist_train_b.GetYaxis().SetLabelSize(0.04)
    hist_train_b.GetYaxis().SetTitleSize(0.04)
    hist_train_b.GetYaxis().SetTitle('Event Fraction')
    hist_train_b.GetXaxis().SetLabelSize(0.04)
    hist_train_b.GetXaxis().SetTitleSize(0.04)
    hist_train_b.GetXaxis().SetTitle('Model Output')
    hist_train_b.SetLineColor(ROOT.kRed)
    hist_train_b.SetLineWidth(3)
    hist_train_b.Draw('HIST')

    hist_test_b.SetMarkerSize(1.3)
    hist_test_b.SetMarkerStyle(3)
    hist_test_b.Draw('same')

    hist_train_s.SetLineColor(ROOT.kBlue)
    hist_train_s.SetLineWidth(3)
    hist_train_s.Draw('hist same')

    hist_test_s.SetMarkerSize(1.3)
    hist_test_s.SetMarkerStyle(8)
    hist_test_s.Draw('same')

    c.SetLogy()

    #Add legend:
    legend = ROOT.TLegend(0.52, 0.75, 0.92, 0.9)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetTextAlign(12)
    legend.AddEntry(hist_train_s, "Signal (Training)", "lf")
    legend.AddEntry(hist_test_s, "Signal (Test)", "pe")
    legend.AddEntry(hist_train_b, "Background (Training)" ,"l")
    legend.AddEntry(hist_test_b, "Background (Test)", "ep")
    legend.Draw("SAME")

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.04)
    text.DrawLatex(0.23, 0.87, "Simulation")
    text.DrawLatex(0.23, 0.83, "H #rightarrow #tau^{+}#tau^{-}")
    text.DrawLatex(0.23, 0.79, "#sqrt{s} = 8 TeV")

    c.Draw()

    #Set marker:
    marker_types = ROOT.TCanvas('marker_types', '', 0,0,500,200)
    marker = ROOT.TMarker()
    marker.DisplayMarkerTypes()
    marker_types.Draw()


    #====================================================
    #     NOW THE WEIGHTED DISTRIBUTION
    #====================================================
    c2 = ROOT.TCanvas("c2", "", 750, 700)

    bins = 10
    hist_train_sw = ROOT.TH1D("hist_train_sw", "train signal", bins, 0, 1)
    hist_train_bw = ROOT.TH1D("hist_train_bw", "train bkg", bins, 0, 1)
    hist_test_w = ROOT.TH1D("hist_test_w", "test bkg", bins, 0, 1)

    for i in range(len(train_signal)):
        hist_train_sw.Fill(train_signal.values[i], weights_train_signal.values[i])
    for i in range(len(train_bkg)):
        hist_train_bw.Fill(train_bkg.values[i], weights_train_bkg.values[i])
    for i in range(len(predictions_test)):
        hist_test_w.Fill(predictions_test.values[i], weights_test.values[i])

    for hist in [hist_train_sw, hist_train_bw, hist_test_w]:
        for i in range(1, hist.GetNbinsX()+1):
            hist.SetBinError(i, np.sqrt(hist.GetBinContent(i)))

    hist_train_sw.SetFillColorAlpha(ROOT.kAzure-1,.6)
    hist_train_bw.SetFillColorAlpha(ROOT.kRed-4, .9)
    hist_train_sw.SetLineWidth(1)
    hist_train_bw.SetLineWidth(1)

    #Axes
    hist_train_bw.GetYaxis().SetLabelSize(0.04)
    hist_train_bw.GetYaxis().SetTitleSize(0.04)
    hist_train_bw.GetYaxis().SetTitle('Events')
    hist_train_bw.GetXaxis().SetLabelSize(0.04)
    hist_train_bw.GetXaxis().SetTitleSize(0.04)
    hist_train_bw.GetXaxis().SetTitle('Model Output')
    hist_train_bw.Draw()

    #Stack
    hs = ROOT.THStack("hs", "Weighted Distributions")
    hs.Add(hist_train_sw)
    hs.Add(hist_train_bw)
    hs.SetMinimum(20)
    hs.SetMaximum(1e7)
    hs.Draw('hist')
    hs.SetHistogram(hist_train_bw)

    hist_test_w.Draw('same')

    #Legend
    legend = ROOT.TLegend(0.5, 0.75, 0.8, 0.9)
    legend.SetTextFont(42)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetTextAlign(12)
    legend.AddEntry(hist_train_sw, "Signal (Training)", "f")
    legend.AddEntry(hist_train_bw, "Background (Training)", "f")
    legend.AddEntry(hist_test_w, "Test", "pe")
    legend.Draw("SAME")

    #Text
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.04)
    text.DrawLatex(0.23, 0.87, "Simulation")
    text.DrawLatex(0.23, 0.83, "H #rightarrow #tau^{+}#tau^{-}")
    text.DrawLatex(0.23, 0.79, "#sqrt{s} = 8 TeV")

    c2.SetLogy()
    c2.Draw()


    #====================================================
    #     SAVE CANVAS
    #====================================================
    c2.SaveAs('weighted.png')
    c2.SaveAs('weighted.pdf')

    w = ROOT.TColorWheel()
    cw = ROOT.TCanvas("cw","cw",0,0,800,800)
    w.SetCanvas(cw)
    w.Draw()
    cw.Draw()


    #====================================================
    #     RATIO PLOT
    #====================================================
    bins = 10

    hist_train_sw = ROOT.TH1D("hist_train_sw", "train signal", bins, 0, 1)
    hist_train_bw = ROOT.TH1D("hist_train_bw", "train bkg", bins, 0, 1)
    hist_test_w = ROOT.TH1D("hist_test_w", "test bkg", bins, 0, 1)

    for i in range(len(train_signal)):
        hist_train_sw.Fill(train_signal.values[i], weights_train_signal.values[i])
    for i in range(len(train_bkg)):
        hist_train_bw.Fill(train_bkg.values[i], weights_train_bkg.values[i])
    for i in range(len(predictions_test)):
        hist_test_w.Fill(predictions_test.values[i], weights_test.values[i])

    for hist in [hist_train_sw, hist_train_bw, hist_test_w]:
        for i in range(1, hist.GetNbinsX()+1):
            hist.SetBinError(i, np.sqrt(hist.GetBinContent(i)))

    c3 = ROOT.TCanvas("c3", "Ratio Plot", 700, 750)

    upper_pad = ROOT.TPad("upper_pad", "", 0, 0.25, 1, 1)
    lower_pad = ROOT.TPad("lower_pad", "", 0, 0, 1, 0.25)
    for pad in [upper_pad, lower_pad]:
        pad.SetLeftMargin(0.14)
        pad.SetRightMargin(0.05)
        pad.SetTickx(True)
        pad.SetTicky(True)
    upper_pad.SetBottomMargin(0)
    lower_pad.SetTopMargin(0)
    lower_pad.SetBottomMargin(0.3)

    upper_pad.Draw()
    lower_pad.Draw()
    c3.Draw()
    
if __name__ == "__main__":
    main()